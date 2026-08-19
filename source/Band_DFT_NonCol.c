/**********************************************************************
  Band_DFT_NonCol.c:

     Band_DFT_NonCol.c is a subroutine to perform band calculations
     based on a non-collinear DFT. 

  Log of Band_DFT_NonCol.c:

     16/Feb./2019  Released by T. Ozaki

***********************************************************************/

#include "mpi.h"
#include "openmx_common.h"
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

#define  measure_time  0

/* GPU workspace for band non-collinear DM (added by H.Kawai, ported from 3.9.9 GPU) */
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

static BandNonColDMGpuWorkspace BandNonCol_dm_gpu_workspace = {0};

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

static BandNonColConstructCache BandNonCol_construct_cache = {0};

static void BandNonCol_AbortWithMessage(const char *msg);

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

static BandNonColGpuSolverWorkspace BandNonCol_gpusolver_workspace = {0};

static void BandNonCol_DMGpu_Destroy(void)
{
    BandNonColDMGpuWorkspace * w = &BandNonCol_dm_gpu_workspace;
    if (w->d_vec_up   != NULL) wait_cudafunc(cudaFree(w->d_vec_up));
    if (w->d_vec_up_e != NULL) wait_cudafunc(cudaFree(w->d_vec_up_e));
    if (w->d_vec_dn   != NULL) wait_cudafunc(cudaFree(w->d_vec_dn));
    if (w->d_vec_dn_e != NULL) wait_cudafunc(cudaFree(w->d_vec_dn_e));
    if (w->d_cols_up  != NULL) wait_cudafunc(cudaFree(w->d_cols_up));
    if (w->d_cols_dn  != NULL) wait_cudafunc(cudaFree(w->d_cols_dn));
    if (w->d_mat11    != NULL) wait_cudafunc(cudaFree(w->d_mat11));
    if (w->d_mat22    != NULL) wait_cudafunc(cudaFree(w->d_mat22));
    if (w->d_mat12    != NULL) wait_cudafunc(cudaFree(w->d_mat12));
    if (w->d_mat11e   != NULL) wait_cudafunc(cudaFree(w->d_mat11e));
    if (w->d_mat22e   != NULL) wait_cudafunc(cudaFree(w->d_mat22e));
    if (w->cublas     != NULL) wait_cudafunc(cublasDestroy(w->cublas));
    if (w->stream     != NULL) wait_cudafunc(cudaStreamDestroy(w->stream));
    memset(w, 0, sizeof(*w));
    w->device_id = -1;
}

static void BandNonCol_GpuSolver_Destroy(void)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;

    if (w->d_A      != NULL) wait_cudafunc(cudaFree(w->d_A));
    if (w->d_W      != NULL) wait_cudafunc(cudaFree(w->d_W));
    if (w->d_info   != NULL) wait_cudafunc(cudaFree(w->d_info));
    if (w->d_work   != NULL) wait_cudafunc(cudaFree(w->d_work));
    if (w->h_work   != NULL) free(w->h_work);
    if (w->gpusolver != NULL) wait_cudafunc(cusolverDnDestroy(w->gpusolver));
    if (w->stream   != NULL) wait_cudafunc(cudaStreamDestroy(w->stream));

    memset(w, 0, sizeof(*w));
    w->device_id = -1;
}

static void BandNonCol_DMGpu_Init(void)
{
    BandNonColDMGpuWorkspace * w = &BandNonCol_dm_gpu_workspace;
    int current_device;
    wait_cudafunc(cudaGetDevice(&current_device));
    if (w->initialized && w->device_id == current_device) return;
    if (w->initialized) BandNonCol_DMGpu_Destroy();
    wait_cudafunc(cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking));
    wait_cudafunc(cublasCreate(&w->cublas));
    wait_cudafunc(cublasSetStream(w->cublas, w->stream));
    w->initialized = 1;
    w->device_id   = current_device;
}

static void BandNonCol_GpuSolver_Init(void)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;
    int current_device;

    wait_cudafunc(cudaGetDevice(&current_device));
    if (w->initialized && w->device_id == current_device) return;
    if (w->initialized) BandNonCol_GpuSolver_Destroy();

    wait_cudafunc(cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking));
    wait_cudafunc(cusolverDnCreate(&w->gpusolver));
    wait_cudafunc(cusolverDnSetStream(w->gpusolver, w->stream));

    w->initialized = 1;
    w->device_id   = current_device;
}

static void BandNonCol_GpuSolver_EnsureInfo(void)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;

    BandNonCol_GpuSolver_Init();
    if (w->d_info==NULL) wait_cudafunc(cudaMalloc((void**)&w->d_info,sizeof(int32_t)));
}

static void BandNonCol_GpuSolver_ReleaseDeviceWorkspace(void)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;

    if (w->d_work!=NULL) wait_cudafunc(cudaFree(w->d_work));
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

static int BandNonCol_EigenvaluesOnlyNoVectors(void)
{
    /* The first (all_knum!=1) diagonalization pass only needs eigenvalues for
       the chemical potential; jobz=CUSOLVER_EIG_MODE_NOVECTOR skips the
       eigenvector build inside cusolverDnXsyevdx and returns identical
       eigenvalues.  Opt out with OPENMX_BAND_SYEVDX_NOVECTOR=0. */
    const char *value = getenv("OPENMX_BAND_SYEVDX_NOVECTOR");

    if (value==NULL){
        return 1;
    }
    return (atoi(value)!=0);
}

static void BandNonCol_ClearOpenAccFreelists(void)
{
#pragma acc wait
    acc_clear_freelists();
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
                                       int turn_mode, int owns_dense_rank, int myid0,
                                       int total_turns);

static int BandNonCol_GpuRequestedMaxTurns(void)
{
    /* By default every k-owner rank that shares the device may solve its
       k-point in the first wave (the plan clamps to the device-group size
       and to the total turn count); the variables only impose an explicit
       upper cap on the concurrency.  Matches the collinear path. */
    const char *value = getenv("OPENMX_BAND_GPU_MAX_CONCURRENT_K");
    long requested = (long)INT_MAX;

    if (value==NULL){
        value = getenv("OPENMX_BAND_GPU_TURN_GROUP");
    }
    if (value==NULL){
        value = getenv("OPENMX_BAND_GPU_MAX_RANKS_PER_DEVICE");
    }
    if (value!=NULL && value[0]!='\0'){
        char *end = NULL;
        long parsed = strtol(value,&end,10);
        if (end!=value && *end=='\0'){
            requested = parsed;
        }
    }
    if (requested<1L){
        requested = 1L;
    }
    if ((long)INT_MAX<requested){
        requested = (long)INT_MAX;
    }
    return (int)requested;
}

static int BandNonCol_GpuTurnGroup(int n, int n2, int MaxN, int size_H1,
                                   int turn_mode, int owns_dense_rank, int myid0,
                                   int total_turns)
{
    return BandNonCol_AutoGpuTurnLimit(BandNonCol_GpuRequestedMaxTurns(),
                                       n,n2,MaxN,size_H1,turn_mode,
                                       owns_dense_rank,myid0,total_turns);
}

static int BandNonCol_MaxConcurrentKGpuTurns(int n, int n2, int MaxN, int size_H1,
                                             int turn_mode, int owns_dense_rank, int myid0,
                                             int total_turns)
{
    return BandNonCol_AutoGpuTurnLimit(BandNonCol_GpuRequestedMaxTurns(),
                                       n,n2,MaxN,size_H1,turn_mode,
                                       owns_dense_rank,myid0,total_turns);
}

static void BandNonCol_GpuSolver_EnsureMatrixCapacity(int n)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;
    size_t nn;

    if (n<=0) BandNonCol_AbortWithMessage("Invalid GpuSolver matrix size in Band_DFT_NonCol.c.");

    BandNonCol_GpuSolver_Init();
    if (n<=w->matrix_dim) return;

    if (w->d_A    != NULL) wait_cudafunc(cudaFree(w->d_A));
    if (w->d_W    != NULL) wait_cudafunc(cudaFree(w->d_W));
    if (w->d_info != NULL) wait_cudafunc(cudaFree(w->d_info));

    nn = (size_t)n*(size_t)n;
    wait_cudafunc(cudaMalloc((void**)&w->d_A,    sizeof(dcomplex)*nn));
    wait_cudafunc(cudaMalloc((void**)&w->d_W,    sizeof(double)*(size_t)n));
    wait_cudafunc(cudaMalloc((void**)&w->d_info, sizeof(int32_t)));

    w->matrix_dim = n;
}

static void BandNonCol_GpuSolver_EnsureWorkspace(int n, int maxn)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;
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

    BandNonCol_GpuSolver_EnsureMatrixCapacity(n);
    range = (n==maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;

    wait_cudafunc(cusolverDnXsyevdx_bufferSize(w->gpusolver,NULL,jobz,range,uplo,n,
                                               CUDA_C_64F,(cuDoubleComplex*)w->d_A,n,&vl,&vu,1L,maxn,&h_meig,
                                               CUDA_R_64F,w->d_W,CUDA_C_64F,&d_bytes,&h_bytes));

    if (w->d_work_bytes<d_bytes){
        if (w->d_work!=NULL) wait_cudafunc(cudaFree(w->d_work));
        w->d_work = NULL;
        if (0<d_bytes) wait_cudafunc(cudaMalloc((void**)&w->d_work,d_bytes));
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
        if (w->h_work==NULL) BandNonCol_AbortWithMessage("Failed to allocate GpuSolver host workspace in Band_DFT_NonCol.c.");
        w->h_work_bytes = h_bytes;
    }
}

static void BandNonCol_GpuSolver_DenseZheevx_Device(dcomplex *A, double *ko, int n, int maxn,
                                                   const char *where, int want_vectors)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;
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

    BandNonCol_GpuSolver_EnsureInfo();
    range = (n==maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;

#pragma acc wait
#pragma acc data present(A[0 : nn], ko[0 : n + 1])
#pragma acc host_data use_device(A, ko)
    {
        wait_cudafunc(cusolverDnXsyevdx_bufferSize(w->gpusolver,NULL,jobz,range,uplo,n,
                                                   CUDA_C_64F,(cuDoubleComplex*)A,n,&vl,&vu,1L,maxn,&h_meig,
                                                   CUDA_R_64F,ko+1,CUDA_C_64F,&d_bytes,&h_bytes));

        if (w->d_work_bytes<d_bytes){
            if (w->d_work!=NULL) wait_cudafunc(cudaFree(w->d_work));
            w->d_work = NULL;
            if (0<d_bytes) wait_cudafunc(cudaMalloc((void**)&w->d_work,d_bytes));
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
            if (w->h_work==NULL) BandNonCol_AbortWithMessage("Failed to allocate GpuSolver host workspace in Band_DFT_NonCol.c.");
            w->h_work_bytes = h_bytes;
        }

        OpenMX_Manifest_Count(MANI_DENSE_GPU_SOLVES);
        if (jobz==CUSOLVER_EIG_MODE_NOVECTOR) OpenMX_Manifest_Count(MANI_DENSE_GPU_NOVECTOR);
        wait_cudafunc(cusolverDnXsyevdx(w->gpusolver,NULL,jobz,range,uplo,n,
                                        CUDA_C_64F,(cuDoubleComplex*)A,n,&vl,&vu,1L,maxn,&h_meig,
                                        CUDA_R_64F,ko+1,CUDA_C_64F,
                                        w->d_work,w->d_work_bytes,w->h_work,w->h_work_bytes,w->d_info));
        wait_cudafunc(cudaMemcpyAsync(&info,w->d_info,sizeof(int32_t),cudaMemcpyDeviceToHost,w->stream));
        wait_cudafunc(cudaStreamSynchronize(w->stream));
    }

    BandNonCol_GpuSolver_ReleaseDeviceWorkspace();

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

static void BandNonCol_GEMMul8Zgemm_OpenACC(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                            dcomplex const * A, dcomplex const * B, dcomplex * C)
{
    const int lda = (transa==CUBLAS_OP_N) ? m : k;
    const int ldb = (transb==CUBLAS_OP_N) ? k : n;

    BandNonCol_DMGpu_Init();
#pragma acc wait
#pragma acc data      present(A[0 : lda * ((transa==CUBLAS_OP_N) ? k : m)], \
                              B[0 : ldb * ((transb==CUBLAS_OP_N) ? n : k)], C[0 : m * n])
#pragma acc host_data use_device(A, B, C)
    {
        cuDoubleComplex const alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex const beta  = make_cuDoubleComplex(0.0, 0.0);
        wait_cudafunc(openmx_gemmul8Zgemm(BandNonCol_dm_gpu_workspace.cublas, transa, transb, m, n, k, &alpha,
                                          (cuDoubleComplex const *)A, lda,
                                          (cuDoubleComplex const *)B, ldb,
                                          &beta,
                                          (cuDoubleComplex *)C, m));
        wait_cudafunc(cudaStreamSynchronize(BandNonCol_dm_gpu_workspace.stream));
    }
}

/* Shared verdict of BandNonCol_GpuDiagFits, consulted by every dense-GPU
   branch so all helpers take the same GPU-vs-CPU path as the flags chosen at
   the top of Band_DFT_NonCol. */
static int BandNonCol_gpu_dense_verdict = 1;

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
    }
    return cached;
}

static int BandNonCol_UseDenseGpuMatrix(int n, int n2)
{
    return (scf_eigen_lib_flag == GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum() <= n2 &&
            na_rows == n && na_cols == n && na_rows2 == n2 && na_cols2 == n2 &&
            BandNonCol_gpu_dense_verdict);
}

static void BandNonCol_DenseTripleTransform_OpenACC(int n, dcomplex *A, dcomplex *S, dcomplex *Work)
{
    size_t nn = (size_t)n * (size_t)n;

#pragma acc data copy(A[0 : nn]) copyin(S[0 : nn]) create(Work[0 : nn])
    {
#pragma acc parallel loop
        for (size_t idx = 0; idx < nn; idx++) {
            Work[idx].r = 0.0;
            Work[idx].i = 0.0;
        }

        BandNonCol_GEMMul8Zgemm_OpenACC(CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, A, S, Work);

#pragma acc parallel loop
        for (size_t idx = 0; idx < nn; idx++) {
            A[idx].r = 0.0;
            A[idx].i = 0.0;
        }

        BandNonCol_GEMMul8Zgemm_OpenACC(CUBLAS_OP_C, CUBLAS_OP_N, n, n, n, S, Work, A);
    }
#pragma acc wait
}

static void BandNonCol_DenseTripleTransform_PresentOpenACC(int n, dcomplex *A, dcomplex *S, dcomplex *Work)
{
    int nn = n*n;

#pragma acc parallel loop present(Work[0 : nn])
    for (int idx=0; idx<nn; idx++){
        Work[idx].r = 0.0;
        Work[idx].i = 0.0;
    }

    BandNonCol_GEMMul8Zgemm_OpenACC(CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,A,S,Work);

#pragma acc parallel loop present(A[0 : nn])
    for (int idx=0; idx<nn; idx++){
        A[idx].r = 0.0;
        A[idx].i = 0.0;
    }

    BandNonCol_GEMMul8Zgemm_OpenACC(CUBLAS_OP_C,CUBLAS_OP_N,n,n,n,S,Work,A);

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

static void BandNonCol_DenseWavefunctions_OpenACC(int n2, dcomplex *Cs2, dcomplex *Ss2, dcomplex *Hs2)
{
    size_t nn = (size_t)n2 * (size_t)n2;

#pragma acc data copyin(Cs2[0 : nn], Ss2[0 : nn]) copyout(Hs2[0 : nn])
    {
#pragma acc parallel loop
        for (size_t idx = 0; idx < nn; idx++) {
            Hs2[idx].r = 0.0;
            Hs2[idx].i = 0.0;
        }

        BandNonCol_GEMMul8Zgemm_OpenACC(CUBLAS_OP_T, CUBLAS_OP_T, n2, n2, n2, Cs2, Ss2, Hs2);
    }
#pragma acc wait
}

static void BandNonCol_DenseWavefunctions_PresentOpenACC(int n2, dcomplex *Cs2, dcomplex *Ss2, dcomplex *Hs2)
{
    int nn = n2*n2;

    if (!acc_is_present(Hs2,sizeof(dcomplex)*(size_t)nn)){
#pragma acc enter data create(Hs2[0 : nn])
    }

#pragma acc parallel loop present(Hs2[0 : nn])
    for (int idx=0; idx<nn; idx++){
        Hs2[idx].r = 0.0;
        Hs2[idx].i = 0.0;
    }

    BandNonCol_GEMMul8Zgemm_OpenACC(CUBLAS_OP_T,CUBLAS_OP_T,n2,n2,n2,Cs2,Ss2,Hs2);

#pragma acc wait
}

static void BandNonCol_AddDense_OpenACC(int n, dcomplex *dst, const dcomplex *src)
{
    int nn = n*n;

#pragma acc parallel loop present(dst[0 : nn], src[0 : nn])
    for (int idx=0; idx<nn; idx++){
        dst[idx].r += src[idx].r;
        dst[idx].i += src[idx].i;
    }
}

static void BandNonCol_GpuSolver_DenseZheevx(dcomplex *A, dcomplex *Z, double *ko, int n, int maxn,
                                            const char *where)
{
    BandNonColGpuSolverWorkspace * w = &BandNonCol_gpusolver_workspace;
    cusolverEigMode_t  jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t   uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range;
    double vl = 0.0;
    double vu = 0.0;
    int64_t h_meig = 0;
    int32_t info = 0;
    int copy_cols;
    size_t nn;

    BandNonCol_GpuSolver_EnsureWorkspace(n,maxn);
    range = (n==maxn) ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;
    nn = (size_t)n*(size_t)n;

    wait_cudafunc(cudaMemcpyAsync(w->d_A,A,sizeof(dcomplex)*nn,cudaMemcpyHostToDevice,w->stream));

    OpenMX_Manifest_Count(MANI_DENSE_GPU_SOLVES);
    if (jobz==CUSOLVER_EIG_MODE_NOVECTOR) OpenMX_Manifest_Count(MANI_DENSE_GPU_NOVECTOR);
    wait_cudafunc(cusolverDnXsyevdx(w->gpusolver,NULL,jobz,range,uplo,n,
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

static size_t BandNonCol_QueryGpuSolverWorkBytes(int n, int maxn)
{
    /* The query costs a cusolver handle + stream create/destroy each time and
       runs several times per SCF iteration; memoize the (n, n)/(n2, MaxN)
       shapes the band path asks about. */
    static int    cached_n[2]     = {-1,-1};
    static int    cached_maxn[2]  = {-1,-1};
    static size_t cached_bytes[2] = {0U,0U};
    static int    cached_next     = 0;
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
        if (cached_n[i]==n && cached_maxn[i]==maxn && cached_bytes[i]!=0U){
            return cached_bytes[i];
        }
    }

    cuda_status = cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    if (cuda_status!=cudaSuccess) return fallback;

    solver_status = cusolverDnCreate(&handle);
    if (solver_status==CUSOLVER_STATUS_SUCCESS){
        solver_status = cusolverDnSetStream(handle,stream);
    }

    if (solver_status==CUSOLVER_STATUS_SUCCESS){
        solver_status = cusolverDnXsyevdx_bufferSize(handle,NULL,jobz,range,uplo,n,
                                                     CUDA_C_64F,NULL,n,&vl,&vu,1L,maxn,&h_meig,
                                                     CUDA_R_64F,NULL,CUDA_C_64F,&d_bytes,&h_bytes);
    }

    if (handle!=NULL) cusolverDnDestroy(handle);
    cudaStreamDestroy(stream);

    if (solver_status!=CUSOLVER_STATUS_SUCCESS || d_bytes==0) return fallback;

    cached_n[cached_next]     = n;
    cached_maxn[cached_next]  = maxn;
    cached_bytes[cached_next] = d_bytes;
    cached_next               = (cached_next + 1) % 2;
    return d_bytes;
}

static size_t BandNonCol_GpuRankOverheadBytes(void)
{
    /* CUDA context, cublas/cusolver handles, and OpenACC pool slack that sit
       outside the explicitly estimated buffers; matches the collinear path
       (OPENMX_BAND_GPU_RANK_OVERHEAD_MB, default 256 MiB per owner rank). */
    const char *value = getenv("OPENMX_BAND_GPU_RANK_OVERHEAD_MB");
    unsigned long long mib = 256ULL;

    if (value!=NULL && value[0]!='\0'){
        char *end = NULL;
        unsigned long long parsed = strtoull(value,&end,10);
        if (end!=value && *end=='\0'){
            mib = parsed;
        }
    }
    if ((unsigned long long)SIZE_MAX/(1024ULL*1024ULL)<mib){
        return SIZE_MAX;
    }
    return (size_t)mib*1024ULL*1024ULL;
}

static size_t BandNonCol_RootDenseDeviceBytes(int n, int n2, int MaxN, int size_H1, int need_evec)
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
    size_t work_n = BandNonCol_QueryGpuSolverWorkBytes(n,n);
    size_t work_n2 = BandNonCol_QueryGpuSolverWorkBytes(n2,MaxN);
    size_t ko_bytes = BandNonCol_ArrayBytes((size_t)n2+1U,sizeof(double),"root dense eigenvalue vector");
    /* The GEMMul8 workspace is a single per-rank buffer that persists across
       the turns of one concurrency group; the eigenvector back-transform is
       an n2-sized GEMM, the triple transforms are n-sized. */
    size_t gemmul8_bytes = need_evec ? openmx_gemmul8ZWorkspaceSize(n2,n2,n2)
                                     : openmx_gemmul8ZWorkspaceSize(n,n,n);
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
                            BandNonCol_CheckedMul(matrix2_bytes,3U,"root dense wavefunction n2*n2 matrices"),
                            "root dense wavefunction n2*n2 matrices");
    }

    peak = BandNonCol_MaxBytes(overlap_peak,BandNonCol_MaxBytes(hamiltonian_peak,wavefunction_peak));
    peak = BandNonCol_CheckedAdd(peak,gemmul8_bytes,"root dense GEMMul8 workspace");
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
    const char *value = getenv("OPENMX_BAND_GPU_RESERVE_MB");

    if (reserve<minimum) reserve = minimum;
    if (value!=NULL && value[0]!='\0'){
        char *end = NULL;
        unsigned long long mib = strtoull(value,&end,10);
        if (end!=value && *end=='\0' &&
            mib<=(unsigned long long)SIZE_MAX/(1024ULL*1024ULL)){
            size_t requested = (size_t)mib*1024ULL*1024ULL;
            if (reserve<requested) reserve = requested;
        }
    }
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
   to the GPU path would abort or spin inside a full device, so the whole band
   phase falls back to the ScaLAPACK/ELPA branch instead.  The verdict is
   collective over mpi_comm_level1 (the branch controls barrier and
   communicator structure) and sticky until the next SCF restart or a change
   of the matrix size.
   OPENMX_BAND_GPU_DIAG=1 forces the GPU path, =0 forces the fallback. */
static int BandNonCol_GpuDiagFits(int SCF_iter, int n, int n2, int MaxN, int size_H1, int myid0)
{
    static int verdict = -1;
    static int verdict_n2 = 0;
    const char *force = getenv("OPENMX_BAND_GPU_DIAG");
    unsigned long long my_required = 0ULL, req_max = 0ULL;
    unsigned long long my_free = ULLONG_MAX, free_min = 0ULL;
    int my_fit = 1;
    int fit = 0;

    if (force!=NULL){
        static int force_announced = 0;
        int forced = (atoi(force)!=0);

        if (!forced && !force_announced && myid0==Host_ID){
            printf("<Band_DFT_NonCol> GPU dense diagonalization disabled by OPENMX_BAND_GPU_DIAG=0.\n");
            fflush(stdout);
        }
        if (!forced) OpenMX_Manifest_SetMax(MANI_DENSE_FB_ENV_OFF, 1LL);   /* B25 */
        force_announced = 1;
        BandNonCol_gpu_dense_verdict = forced;
        return forced;
    }

    if (verdict!=-1 && verdict_n2==n2 && 1<SCF_iter){
        BandNonCol_gpu_dense_verdict = verdict;
        return verdict;
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
                BandNonCol_RootDenseDeviceBytes(n,n2,MaxN,size_H1,1),
                BandNonCol_RootDenseDMDeviceBytes(n,n2,MaxN,size_H1));
            size_t free_bytes = 0U, total_bytes = 0U;

            if (required==0U || required==SIZE_MAX ||
                cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess){
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

    if (fit==0 && myid0==Host_ID){
        OpenMX_Manifest_Count(MANI_DENSE_FB_MEM_EVENTS);   /* B26 */
        printf("<Band_DFT_NonCol> A k-owner rank cannot fit even one k-point's dense GPU"
               " diagonalization (%.1f MiB needed per rank, %.1f MiB free);"
               " falling back to the ScaLAPACK/ELPA diagonalization."
               " Force the GPU path with OPENMX_BAND_GPU_DIAG=1 or lower"
               " OPENMX_BAND_GPU_RESERVE_MB.\n",
               (double)req_max/(1024.0*1024.0),
               (free_min==ULLONG_MAX) ? 0.0 : (double)free_min/(1024.0*1024.0));
        fflush(stdout);
    }

    verdict = fit;
    verdict_n2 = n2;
    BandNonCol_gpu_dense_verdict = fit;
    return verdict;
}

static int BandNonCol_AutoGpuTurnLimit(int requested, int n, int n2, int MaxN, int size_H1,
                                       int turn_mode, int owns_dense_rank, int myid0,
                                       int total_turns)
{
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
                : BandNonCol_RootDenseDeviceBytes(n,n2,MaxN,size_H1,
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
            cuda_status = cudaMemGetInfo(&free_bytes,&total_bytes);
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
                static const char *need_name[3] =
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
                if (device_rank==0){
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

            if (device_rank==0){
                static int last_limit[3] = {-1,-1,-1};
                static int last_device_ranks[3] = {-1,-1,-1};
                static const char *mode_label[3] =
                    {"eigenvalues only","eigenvectors","density matrix"};
                int mode_slot = (0<=turn_mode && turn_mode<3) ? turn_mode : 1;

                OpenMX_Manifest_SetMax(MANI_DIAG_CONCURRENCY_PLAN, (long long)selected_limit);   /* B27 */
                OpenMX_Manifest_SetMin(MANI_DIAG_CONCURRENCY_MIN,  (long long)selected_limit);
                if (last_limit[mode_slot]!=selected_limit ||
                    last_device_ranks[mode_slot]!=device_ranks){
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
                    last_limit[mode_slot] = selected_limit;
                    last_device_ranks[mode_slot] = device_ranks;
                }
            }
        }
        else {
            local_limit = 1;
            if (device_rank==0){
                OpenMX_Manifest_SetMin(MANI_DIAG_CONCURRENCY_MIN, 1LL);
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

static int BandNonCol_RootDenseParallelKWorldsFit(int n, int n2, int MaxN, int size_H1,
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
        if (myid0==Comm_World_StartID2[myworld2]){
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
            required_bytes = BandNonCol_RootDenseDeviceBytes(n,n2,MaxN,size_H1,1);
            cuda_status = cudaMemGetInfo(&free_bytes,&total_bytes);
        }

        if (cuda_status==cudaSuccess){
            reserve_bytes = BandNonCol_RootDenseReserveBytes(total_bytes);
            cuda_ok = 1;
        }
        else {
            fprintf(stderr,
                    "<Band>  Rank %d failed to query CUDA memory for non-collinear dense GpuSolver (%s); "
                    "using serialized k-worlds.\n",
                    myid0,cudaGetErrorString(cuda_status));
            fflush(stderr);
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
            if (device_rank==0){
                OpenMX_Manifest_Count(MANI_BAND_KWORLD_SERIALIZED);   /* B28 */
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
#pragma acc exit data delete(entries[0 : dense_count])
        }
        if (phase_r!=NULL && phase_i!=NULL && 0<phase_count){
#pragma acc exit data delete(phase_r[0 : phase_count], phase_i[0 : phase_count])
        }
        BandNonCol_ClearOpenAccFreelists();
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

static void BandNonCol_ConstructCache_EnsureDenseDevice(void)
{
    BandNonColConstructCache *cache = &BandNonCol_construct_cache;

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

static void BandNonCol_ConstructDenseMs_OpenACC(int cpx_flag, int n, double k1, double k2, double k3,
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
#pragma acc update device(phase_r[0 : phase_count], phase_i[0 : phase_count])
    }

    if (!acc_is_present(Ms,sizeof(dcomplex)*(size_t)matrix_count)){
#pragma acc enter data create(Ms[0 : matrix_count])
    }

#pragma acc data copyin(M2[0 : h_count]) present(Ms[0 : matrix_count], entries[0 : dense_count], phase_r[0 : phase_count], phase_i[0 : phase_count])
    {
#pragma acc parallel loop
        for (int idx=0; idx<matrix_count; idx++){
            Ms[idx].r = 0.0;
            Ms[idx].i = 0.0;
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

static void BandNonCol_AccumulateDMKPoint_OpenACC(int myid2, int *is2, int *ie2, int *MP,
                                                  int n, int n2, int size_H1,
                                                  double k1, double k2, double k3, const double *ko,
                                                  const dcomplex *EVec1,
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
    const dcomplex *evec_ptr = EVec1;
    int basis_stride = ie2[myid2] - is2[myid2] + 1;
    int nk;
    int entry_count;
    int pair_count;
    size_t evec_count;

    if (basis_stride<=0) return;

    BandNonCol_DMOccWorkspace_Ensure(basis_stride);
    nk = BandNonCol_BuildOccupationWeightsDense(is2[myid2],ie2[myid2],ko,ws->occ,ws->eig_occ);

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
    evec_count = (size_t)n2*(size_t)basis_stride;

#pragma acc data copyin(evec_ptr[0:evec_count], phase_r[0:pair_count], phase_i[0:pair_count], occ[0:nk], eig_occ[0:nk])
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

/* "copy" (not "copyout") + "+=" below: the kernel ACCUMULATES on top of the
   host arrays, so a caller may invoke it once per k point; the single-k
   callers zero their arrays first and keep the old behavior */
#pragma acc data copyin(basis0_chunk[0:chunk_count], basis1_chunk[0:chunk_count], phase_index_chunk[0:chunk_count]) \
                 copy(rDM11_chunk[0:chunk_count], rDM22_chunk[0:chunk_count], rDM12_chunk[0:chunk_count], \
                      iDM12_chunk[0:chunk_count], iDM11_chunk[0:chunk_count], iDM22_chunk[0:chunk_count], \
                      rEDM11_chunk[0:chunk_count], rEDM22_chunk[0:chunk_count])
            {
#pragma acc parallel loop gang present(evec_ptr[0:evec_count], phase_r[0:pair_count], phase_i[0:pair_count], \
                                       occ[0:nk], eig_occ[0:nk])
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

#pragma acc loop seq
                    for (int k=0; k<nk; k++){
                        const double w = occ[k];
                        const double ew = eig_occ[k];
                        const dcomplex va_up = evec_ptr[(size_t)ia*(size_t)basis_stride + (size_t)k];
                        const dcomplex vb_up = evec_ptr[(size_t)ib*(size_t)basis_stride + (size_t)k];
                        const dcomplex va_dn = evec_ptr[(size_t)(ia+n)*(size_t)basis_stride + (size_t)k];
                        const dcomplex vb_dn = evec_ptr[(size_t)(ib+n)*(size_t)basis_stride + (size_t)k];
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

    /* the copyin of the eigenvector panel aborts instead of failing softly,
       so refuse up front when it clearly does not fit */
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return (evec_bytes + 256ULL*1024ULL*1024ULL <= free_bytes);
}

static void BandNonCol_AccumulateDMRootDense_OpenACC(int *MP, int n, int n2, int max_state, int size_H1,
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

#pragma acc data copyin(dense_evec[0:evec_count], phase_r[0:pair_count], phase_i[0:pair_count], occ[0:nk], eig_occ[0:nk])
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

/* "copy" (not "copyout") + "+=" below: the kernel ACCUMULATES on top of the
   host arrays, so a caller may invoke it once per k point; the single-k
   callers zero their arrays first and keep the old behavior */
#pragma acc data copyin(basis0_chunk[0:chunk_count], basis1_chunk[0:chunk_count], phase_index_chunk[0:chunk_count]) \
                 copy(rDM11_chunk[0:chunk_count], rDM22_chunk[0:chunk_count], rDM12_chunk[0:chunk_count], \
                      iDM12_chunk[0:chunk_count], iDM11_chunk[0:chunk_count], iDM22_chunk[0:chunk_count], \
                      rEDM11_chunk[0:chunk_count], rEDM22_chunk[0:chunk_count])
            {
#pragma acc parallel loop gang present(dense_evec[0:evec_count], phase_r[0:pair_count], phase_i[0:pair_count], \
                                       occ[0:nk], eig_occ[0:nk])
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

#pragma acc loop seq
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

static void BandNonCol_AccumulateDMRootDenseK_OpenACC(int size_H1, int *MP, int n, int n2, int MaxN,
                                                      double k1, double k2, double k3,
                                                      double *ko, dcomplex *dense_evec,
                                                      double *rDM11, double *rDM22, double *rDM12,
                                                      double *iDM12, double *iDM11, double *iDM22,
                                                      double *rEDM11, double *rEDM22)
{
    double *dm_buffer;

    if (dense_evec==NULL) return;

    dm_buffer = (double*)malloc(sizeof(double)*(size_t)size_H1*8U);
    if (dm_buffer==NULL){
        BandNonCol_AbortWithMessage("Failed to allocate BandNonCol OpenACC DM accumulation buffer.");
    }
    memset(dm_buffer,0,sizeof(double)*(size_t)size_H1*8U);

    BandNonCol_AccumulateDMRootDense_OpenACC(MP,n,n2,MaxN,size_H1,k1,k2,k3,ko,dense_evec,
                                             dm_buffer + (size_t)size_H1*0U,
                                             dm_buffer + (size_t)size_H1*1U,
                                             dm_buffer + (size_t)size_H1*2U,
                                             dm_buffer + (size_t)size_H1*3U,
                                             dm_buffer + (size_t)size_H1*4U,
                                             dm_buffer + (size_t)size_H1*5U,
                                             dm_buffer + (size_t)size_H1*6U,
                                             dm_buffer + (size_t)size_H1*7U);

    for (int p=0; p<size_H1; p++){
        rDM11[p]  += dm_buffer[(size_t)size_H1*0U + (size_t)p];
        rDM22[p]  += dm_buffer[(size_t)size_H1*1U + (size_t)p];
        rDM12[p]  += dm_buffer[(size_t)size_H1*2U + (size_t)p];
        iDM12[p]  += dm_buffer[(size_t)size_H1*3U + (size_t)p];
        iDM11[p]  += dm_buffer[(size_t)size_H1*4U + (size_t)p];
        iDM22[p]  += dm_buffer[(size_t)size_H1*5U + (size_t)p];
        rEDM11[p] += dm_buffer[(size_t)size_H1*6U + (size_t)p];
        rEDM22[p] += dm_buffer[(size_t)size_H1*7U + (size_t)p];
    }

    free(dm_buffer);
}

static void BandNonCol_CalcDMAllK1_OpenACC(int myid0, int myid2, int size_H1,
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

    BandNonCol_AccumulateDMKPoint_OpenACC(myid2,is2,ie2,MP,n,n2,size_H1,k1,k2,k3,ko,EVec1,
                                          rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,rEDM11,rEDM22);

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
    dcomplex *ss2;
    dcomplex *cs2;
} BandNonColRootDenseWorkspace;

static BandNonColRootDenseWorkspace BandNonCol_root_dense_workspace = {0};

static void BandNonCol_RootDenseWorkspace_Reset(void)
{
    BandNonColRootDenseWorkspace *ws = &BandNonCol_root_dense_workspace;

    if (ws->cs2_on_device && ws->cs2 != NULL){
        dcomplex *cs2 = ws->cs2;
        size_t n2n2 = (size_t)ws->n2*(size_t)ws->n2;

#pragma acc exit data delete(cs2[0 : n2n2])
    }
    free(ws->s_all);
    free(ws->h11);
    free(ws->h22);
    free(ws->h12);
    free(ws->work);
    free(ws->hs2);
    free(ws->ss2);
    free(ws->cs2);
    memset(ws,0,sizeof(*ws));
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
        ws->ss2   = (dcomplex*)malloc(sizeof(dcomplex)*n2n2);
        ws->cs2   = (dcomplex*)malloc(sizeof(dcomplex)*n2n2);

        if (ws->s_all==NULL || ws->h11==NULL || ws->h22==NULL || ws->h12==NULL ||
            ws->work==NULL || ws->hs2==NULL || ws->ss2==NULL || ws->cs2==NULL){
            BandNonCol_RootDenseWorkspace_Reset();
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

static void BandNonCol_ConstructDenseMsFromPacked( int cpx_flag, const double *M1, dcomplex *Ms,
                                                   int *order_GA, int *MP, double k1, double k2, double k3,
                                                   int n, int owns_dense );
static void BandNonCol_BuildDenseSs2_OpenACC(int n, int n2, const dcomplex *S, dcomplex *S2);
static void BandNonCol_BuildDenseHs2_OpenACC(int n, int n2, const dcomplex *H11,
                                             const dcomplex *H22, const dcomplex *H12,
                                             dcomplex *H2);

static void BandNonCol_RootDenseSolveOneK_OpenACC(int rebuild_overlap,
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
    int nn = n*n;
    int n2n2 = n2*n2;
    int i,j,l;
    dcomplex *active_S = rdw->s_all;

#pragma acc enter data create(ko[0 : n2 + 1])

    if (!rebuild_overlap){
#pragma acc enter data copyin(active_S[0 : nn])
    }

    if (rebuild_overlap){
        BandNonCol_ConstructDenseMsFromPacked(0,m_olp,active_S,packed_order_GA,MP,k1,k2,k3,n,1);

        BandNonCol_SymmetrizeDenseHermitian_OpenACC(n,active_S);
        BandNonCol_GpuSolver_DenseZheevx_Device(active_S,ko,n,n,
                                               "Band_DFT_NonCol root dense overlap",1);

#pragma acc parallel loop present(ko[0 : n + 1])
        for (l=1; l<=n; l++){
            if (ko[l]<1.0e-10) ko[l] = 1.0e-10;
            ko[l] = 1.0/sqrt(ko[l]);
        }

#pragma acc parallel loop collapse(2) present(active_S[0 : nn], ko[0 : n + 1])
        for (i=0; i<n; i++){
            for (j=0; j<n; j++){
                active_S[(size_t)j*(size_t)n + (size_t)i].r *= ko[j+1];
                active_S[(size_t)j*(size_t)n + (size_t)i].i *= ko[j+1];
            }
        }

#pragma acc update self(active_S[0 : nn])
    }

    BandNonCol_ConstructDenseMsFromPacked(0,m_h11,rdw->h11,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_ConstructDenseMsFromPacked(1,m_i11,rdw->work,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_AddDense_OpenACC(n,rdw->h11,rdw->work);

    BandNonCol_ConstructDenseMsFromPacked(0,m_h22,rdw->h22,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_ConstructDenseMsFromPacked(1,m_i22,rdw->work,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_AddDense_OpenACC(n,rdw->h22,rdw->work);

    BandNonCol_ConstructDenseMsFromPacked(0,m_h12,rdw->h12,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_ConstructDenseMsFromPacked(1,m_h12i,rdw->work,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_AddDense_OpenACC(n,rdw->h12,rdw->work);
    BandNonCol_ConstructDenseMsFromPacked(1,m_i12,rdw->work,packed_order_GA,MP,k1,k2,k3,n,1);
    BandNonCol_AddDense_OpenACC(n,rdw->h12,rdw->work);

    {
        dcomplex *h11 = rdw->h11;
        dcomplex *h22 = rdw->h22;
        dcomplex *h12 = rdw->h12;
        dcomplex *work = rdw->work;
        dcomplex *hs2 = rdw->hs2;
        dcomplex *ss2 = rdw->ss2;
        dcomplex *cs2 = rdw->cs2;

        /* The GEMMul8 workspace stays allocated across the turns of one
           concurrency group (the auto turn limit budgets for it) and is
           released in the group-end cleanup of each caller. */
        BandNonCol_DenseTripleTransform_PresentOpenACC(n,h11,active_S,work);
        BandNonCol_DenseTripleTransform_PresentOpenACC(n,h12,active_S,work);
        BandNonCol_DenseTripleTransform_PresentOpenACC(n,h22,active_S,work);

#pragma acc exit data delete(work[0 : nn])
        BandNonCol_BuildDenseHs2_OpenACC(n,n2,h11,h22,h12,hs2);
#pragma acc exit data delete(h11[0 : nn], h22[0 : nn], h12[0 : nn])

        BandNonCol_SymmetrizeDenseHermitian_OpenACC(n2,hs2);
        BandNonCol_GpuSolver_DenseZheevx_Device(hs2,ko,n2,MaxN,
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
            return;
        }

        BandNonCol_BuildDenseSs2_OpenACC(n,n2,active_S,ss2);
#pragma acc exit data delete(active_S[0 : nn])

        BandNonCol_DenseWavefunctions_PresentOpenACC(n2,hs2,ss2,cs2);

        /* cs2 stays device-resident: the 16*n2*n2 host readback used to be
           the biggest host-touched block of a k-owner rank, and both
           consumers can work from the device copy (the scatter packs
           per-destination panels there, the DM copyin turns into a
           present hit).  The copy lives until the workspace reset. */
        rdw->cs2_on_device = 1;
#pragma acc exit data delete(hs2[0 : n2n2], ss2[0 : n2n2])
#pragma acc exit data delete(ko[0 : n2 + 1])
        BandNonCol_ClearOpenAccFreelists();
    }
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
            send_buf = (dcomplex*)malloc(sizeof(dcomplex)*(size_t)max_count);
            if (send_buf==NULL){
                BandNonCol_AbortWithMessage("Failed to allocate dense eigenvector send buffer in Band_DFT_NonCol.c.");
            }
        }

        if (evec_on_device && 0<max_count){
            /* pack each destination's panel on the device: the host never
               materialises the full 16*n2*n2 eigenvector block, only one
               n2 x stride panel at a time */
            panel_dev = (dcomplex*)acc_malloc(sizeof(dcomplex)*(size_t)max_count);
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

#pragma acc parallel loop gang vector_length(128) present(dense_evec[0 : n2n2]) deviceptr(panel)
                for (int basis=0; basis<n2; basis++){
#pragma acc loop vector
                    for (int s=0; s<stride_c; s++){
                        panel[(size_t)basis*(size_t)stride_c + (size_t)s] =
                            dense_evec[(size_t)basis*(size_t)n2 + (size_t)(is_c-1) + (size_t)s];
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

        if (panel_dev!=NULL) acc_free(panel_dev);
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

static void BandNonCol_BuildDenseSs2_OpenACC(int n, int n2, const dcomplex *S, dcomplex *S2)
{
    int nn = n*n;
    int n2n2 = n2*n2;

    if (!acc_is_present(S2,sizeof(dcomplex)*(size_t)n2n2)){
#pragma acc enter data create(S2[0 : n2n2])
    }

#pragma acc parallel loop collapse(2) present(S[0 : nn], S2[0 : n2n2])
    for (int i=0; i<n2; i++){
        for (int j=0; j<n2; j++){
            int idx2 = n2*i + j;

            if (i<n && j<n){
                int idx = n*i + j;
                S2[idx2] = S[idx];
            }
            else if (n<=i && n<=j){
                int idx = n*(i-n) + (j-n);
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
  int root_dense_serial_gpusolver_worlds = 0;
  int owns_dense_k_rank;
  int *dense_k_owner = NULL;
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

	  if (scf_eigen_lib_flag==GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum()<=n2){
	    (void)BandNonCol_GpuDiagFits(SCF_iter,n,n2,MaxN,size_H1,myid0);
	  }
	  else {
	    /* not a GPU-dense run; keep the shared verdict neutral so the helper
	       branches follow their other conditions unchanged */
	    BandNonCol_gpu_dense_verdict = 1;
	  }
	  use_root_dense_gpusolver = (scf_eigen_lib_flag==GPUSOLVER && all_knum==1 && Band_DFT_NonCol_GpuSwitchNum()<=n2 &&
	                              BandNonCol_gpu_dense_verdict);
	  use_k_dense_gpusolver = (scf_eigen_lib_flag==GPUSOLVER && all_knum!=1 &&
	                          Band_DFT_NonCol_GpuSwitchNum()<=n2 && strcasecmp(mode,"scf")==0 &&
	                          BandNonCol_gpu_dense_verdict);
	  OpenMX_Manifest_SetMax(MANI_BAND_ALL_KNUM_P1, (long long)(all_knum+1));
	  OpenMX_Manifest_SetMax(MANI_BAND_T_KNUM, (long long)T_knum);
	  owns_dense_k_rank = (use_k_dense_gpusolver && Set_Hamiltonian_OpenACC_Rank_Is_Selected());
	  if (owns_dense_k_rank || (use_root_dense_gpusolver && Set_Hamiltonian_OpenACC_Rank_Is_Selected()))
	    OpenMX_Manifest_RankFlag(MANI_KOWNER_RANKS);
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

    int max_concurrent_gpu_turns =
      BandNonCol_MaxConcurrentKGpuTurns(n,n2,MaxN,size_H1,
                                        BANDNONCOL_GPU_TURN_EIGVALS,
                                        owns_dense_k_rank,myid0,T_knum);
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

    for (int group_first=0; group_first<T_knum; group_first+=max_concurrent_gpu_turns){
      int group_last =
        (group_first + max_concurrent_gpu_turns < T_knum) ?
        (group_first + max_concurrent_gpu_turns) : T_knum;
      int owns_dense_k_group = 0;

      for (int gpu_turn=group_first; gpu_turn<group_last; gpu_turn++){
        if (dense_k_owner[gpu_turn]==myid0){
          owns_dense_k_group = 1;
          break;
        }
      }

      MPI_Barrier(mpi_comm_level1);

      if (owns_dense_k_group){
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

      for (int gpu_turn=group_first; gpu_turn<group_last; gpu_turn++){

        kloop = gpu_turn;
        if (dense_k_owner[kloop]!=myid0) continue;

        k1 = T_KGrids1[kloop];
        k2 = T_KGrids2[kloop];
        k3 = T_KGrids3[kloop];

        rdw = BandNonCol_RootDenseWorkspace_Ensure(1,n,n2,MaxN,1,SCF_iter);
        BandNonCol_RootDenseSolveOneK_OpenACC(1,n,n2,MaxN,kloop,k1,k2,k3,
                                              m_olp,m_h11,m_h22,m_h12,m_h12i,
                                              m_i11,m_i22,m_i12,
                                              order_GA,MP,ko,EIGEN,0,rdw);
      }

      if (owns_dense_k_group){
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
        if (BandNonCol_GemmWorkspaceTurnRelease()){
          openmx_gemmul8ReleaseWorkspaces();
        }
        BandNonCol_ConstructCache_Reset();
        BandNonCol_RootDenseWorkspace_Reset();
        BandNonCol_GpuSolver_Destroy();
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
      BandNonCol_GpuTurnGroup(n,n2,MaxN,size_H1,
                              BANDNONCOL_GPU_TURN_EIGVECS,
                              owns_root_dense_rank,myid0,Num_Comm_World2) :
      Num_Comm_World2;
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

    if (root_dense_turn_group<1) root_dense_turn_group = 1;
    root_dense_serial_gpusolver_worlds = 0;

    if (use_setham_packed_cache){
      Set_Hamiltonian_GpuSolver_SetMP(MP);
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
          k1 = T_KGrids1[kloop];
          k2 = T_KGrids2[kloop];
          k3 = T_KGrids3[kloop];

          if (owns_root_dense){
            BandNonCol_RootDenseSolveOneK_OpenACC(rebuild_overlap,n,n2,MaxN,kloop,k1,k2,k3,
                                                  m_olp,m_h11,m_h22,m_h12,m_h12i,
                                                  m_i11,m_i22,m_i12,
                                                  packed_order_GA,MP,ko,EIGEN,1,rdw);
          }

          BandNonCol_DistributeDenseEvecGlobal(kloop,n2,MaxN,myid0,myworld2,NPROCS_WD2,
                                               Comm_World_StartID2,root_dense_owner,
                                               owns_root_dense ? rdw->cs2 : NULL,
                                               owns_root_dense ? rdw->cs2_on_device : 0,EVec1[0]);

          if (owns_root_dense) rdw->s_valid = 1;
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
        if (BandNonCol_GemmWorkspaceTurnRelease()){
          openmx_gemmul8ReleaseWorkspaces();
        }
        BandNonCol_ConstructCache_Reset();
        BandNonCol_GpuSolver_Destroy();
      }

      if (serialize_gpu_turns){
        MPI_Barrier(mpi_comm_level1);
      }
    }

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
      Construct_Band_Ms(0,CntOLP,H1,Cs,MP,k1,k2,k3);

      /* diagonalize Cs */

      if (kloop0<num_kloop0){

	MPI_Comm_split(MPI_CommWD2[myworld2],my_pcol,my_prow,&mpi_comm_rows);
	MPI_Comm_split(MPI_CommWD2[myworld2],my_prow,my_pcol,&mpi_comm_cols);

	mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
	mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

	if (scf_eigen_lib_flag==GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum()<=n2 && na_rows==n && na_cols==n &&
	    BandNonCol_gpu_dense_verdict){
	  BandNonCol_GpuSolver_DenseZheevx(Cs,Ss,ko,n,n,"Band_DFT_NonCol overlap");
	}
	else if (scf_eigen_lib_flag==1 || (numprocs2<5 && scf_eigen_lib_flag!=GPUSOLVER)){
	  OpenMX_Manifest_Count(MANI_DENSE_CPU_SOLVES);
	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
	    ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}
	
	else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){
	  OpenMX_Manifest_Count(MANI_DENSE_CPU_SOLVES);

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
        BandNonCol_DenseTripleTransform_OpenACC(n,Hs11,Ss,Cs);
        BandNonCol_DenseTripleTransform_OpenACC(n,Hs12,Ss,Cs);
        BandNonCol_DenseTripleTransform_OpenACC(n,Hs22,Ss,Cs);
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

        if (scf_eigen_lib_flag==GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum()<=n2 && na_rows2==n2 && na_cols2==n2 &&
            BandNonCol_gpu_dense_verdict){
          BandNonCol_GpuSolver_DenseZheevx(Hs2,Cs2,ko,n2,MaxN,"Band_DFT_NonCol Hamiltonian");
        }
        else if (scf_eigen_lib_flag==1 || (numprocs2<5 && scf_eigen_lib_flag!=GPUSOLVER)){
	  OpenMX_Manifest_Count(MANI_DENSE_CPU_SOLVES);
	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}

        else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){
        OpenMX_Manifest_Count(MANI_DENSE_CPU_SOLVES);

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
        BandNonCol_DenseWavefunctions_OpenACC(n2,Cs2,Ss2,Hs2);
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
  if (BandNonCol_UseDenseGpuMatrix(n,n2) && BandNonCol_GemmWorkspaceTurnRelease()){
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
          BandNonCol_GpuTurnGroup(n,n2,MaxN,size_H1,
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
              BandNonColRootDenseWorkspace *rdw = &BandNonCol_root_dense_workspace;

              if (!rdw->valid || rdw->cs2==NULL){
                BandNonCol_AbortWithMessage("Root dense eigenvectors are not available for Band_DFT_NonCol DM.");
              }

              kloop = root_dense_world;
              k1 = T_KGrids1[kloop];
              k2 = T_KGrids2[kloop];
              k3 = T_KGrids3[kloop];

              BandNonCol_AccumulateDMRootDenseK_OpenACC(size_H1,MP,n,n2,MaxN,k1,k2,k3,
                                                        EIGEN[0][kloop],rdw->cs2,
                                                        rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
                                                        rEDM11,rEDM22);
            }
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
        Calc_DM_Band_non_collinear( 0,1,
				    myid0,myid2,size_H1,
				    is2,ie2,MP,n,n2,MaxN,k1,k2,k3,
				    CDM,iDM[0],EDM,EIGEN[0][0],
				    EVec1[0],
				    rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
				    rEDM11,rEDM22 );
      }
      else if (scf_eigen_lib_flag==GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum()<=n2){
        BandNonCol_CalcDMAllK1_OpenACC( myid0,myid2,size_H1,
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
        BandNonCol_MaxConcurrentKGpuTurns(n,n2,MaxN,size_H1,
                                          BANDNONCOL_GPU_TURN_EIGVECS,
                                          owns_dense_k_rank,myid0,T_knum);
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

        for (int gpu_turn=group_first; gpu_turn<group_last; gpu_turn++){
          if (dense_k_owner[gpu_turn]==myid0){
            owns_dense_k_group = 1;
            break;
          }
        }

        MPI_Barrier(mpi_comm_level1);

        if (owns_dense_k_group){
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
        else {
          pack_buffer = (double*)malloc(sizeof(double)*(size_t)size_H1);
          if (pack_buffer==NULL){
            BandNonCol_AbortWithMessage("Failed to allocate dense k-point DM packing buffer in Band_DFT_NonCol.c.");
          }
        }

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

        for (int gpu_turn=group_first; gpu_turn<group_last; gpu_turn++){

          kloop = gpu_turn;
          if (dense_k_owner[kloop]!=myid0) continue;

          k1 = T_KGrids1[kloop];
          k2 = T_KGrids2[kloop];
          k3 = T_KGrids3[kloop];

          rdw = BandNonCol_RootDenseWorkspace_Ensure(1,n,n2,MaxN,1,SCF_iter);
          BandNonCol_RootDenseSolveOneK_OpenACC(1,n,n2,MaxN,kloop,k1,k2,k3,
                                                m_olp,m_h11,m_h22,m_h12,m_h12i,
                                                m_i11,m_i22,m_i12,
                                                order_GA,MP,ko,EIGEN,1,rdw);
          BandNonCol_AccumulateDMRootDenseK_OpenACC(size_H1,MP,n,n2,MaxN,k1,k2,k3,
                                                    EIGEN[0][kloop],rdw->cs2,
                                                    rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
                                                      rEDM11,rEDM22);
        }

        if (owns_dense_k_group){
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
          if (BandNonCol_GemmWorkspaceTurnRelease()){
            openmx_gemmul8ReleaseWorkspaces();
          }
          BandNonCol_ConstructCache_Reset();
          BandNonCol_RootDenseWorkspace_Reset();
          BandNonCol_GpuSolver_Destroy();
        }

        MPI_Barrier(mpi_comm_level1);
      }

      kloop = 0;
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

        if (scf_eigen_lib_flag==GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum()<=n2 && na_rows==n && na_cols==n &&
            BandNonCol_gpu_dense_verdict){
          BandNonCol_GpuSolver_DenseZheevx(Cs,Ss,ko,n,n,"Band_DFT_NonCol overlap");
        }
        else if (scf_eigen_lib_flag==1 || (numprocs2<5 && scf_eigen_lib_flag!=GPUSOLVER)){
  	  OpenMX_Manifest_Count(MANI_DENSE_CPU_SOLVES);
  	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n, &n, Cs, &na_rows, &ko[1], Ss, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}

        else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){
          OpenMX_Manifest_Count(MANI_DENSE_CPU_SOLVES);

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
          BandNonCol_DenseTripleTransform_OpenACC(n,Hs11,Ss,Cs);
          BandNonCol_DenseTripleTransform_OpenACC(n,Hs12,Ss,Cs);
          BandNonCol_DenseTripleTransform_OpenACC(n,Hs22,Ss,Cs);
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
  
        if (scf_eigen_lib_flag==GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum()<=n2 && na_rows2==n2 && na_cols2==n2 &&
            BandNonCol_gpu_dense_verdict){
          BandNonCol_GpuSolver_DenseZheevx(Hs2,Cs2,ko,n2,MaxN,"Band_DFT_NonCol Hamiltonian");
        }
        else if (scf_eigen_lib_flag==1 || (numprocs2<5 && scf_eigen_lib_flag!=GPUSOLVER)){
	  OpenMX_Manifest_Count(MANI_DENSE_CPU_SOLVES);
	  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)
          ( &n2, &MaxN, Hs2, &na_rows2, &ko[1], Cs2, &na_rows2, &nblk2, &mpi_comm_rows_int, &mpi_comm_cols_int );
	}

        else if (scf_eigen_lib_flag==2 || scf_eigen_lib_flag==GPUSOLVER){
          OpenMX_Manifest_Count(MANI_DENSE_CPU_SOLVES);

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
          BandNonCol_DenseWavefunctions_OpenACC(n2,Cs2,Ss2,Hs2);
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

        if (kloop0<num_kloop0 &&
            BandNonCol_UseGpuFallbackDM(n2,(size_t)n2*(size_t)(ie2[myid2]-is2[myid2]+1)*sizeof(dcomplex))){
          /* accumulates this k on the GPU; EVec1 stays unscaled because the
             kernel applies the Fermi weights itself */
          OpenMX_Manifest_Count(MANI_DM_GPU_CALLS);
          BandNonCol_AccumulateDMKPoint_OpenACC(myid2,is2,ie2,MP,n,n2,size_H1,k1,k2,k3,
                                                EIGEN[0][kloop],EVec1[0],
                                                rDM11,rDM22,rDM12,iDM12,iDM11,iDM22,
                                                rEDM11,rEDM22);
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

    /* release the GEMMul8 workspace left behind by the per-k dense GEMMs */
    if (BandNonCol_UseDenseGpuMatrix(n,n2) && BandNonCol_GemmWorkspaceTurnRelease()){
      openmx_gemmul8ReleaseWorkspaces();
    }

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

  if (scf_eigen_lib_flag==GPUSOLVER && Band_DFT_NonCol_GpuSwitchNum()<=2*n){
    BandNonCol_ConstructCache_Ensure(order_GA,MP,n);
    BandNonCol_ConstructDenseMs_OpenACC(cpx_flag,n,k1,k2,k3,M1,Ms);
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
    BandNonCol_ConstructDenseMs_OpenACC(cpx_flag,n,k1,k2,k3,M1,Ms);
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
