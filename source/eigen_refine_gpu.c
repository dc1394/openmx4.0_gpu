/* Refined FP32 eigensolver of the dense GPU paths (see eigen_refine_gpu.h).

   Algorithm (Ogita and Aishima, RefSyEv, with clusters): A is made symmetric
   or Hermitian from its lower triangle, the part the eigensolver reads; all
   n eigenpairs are solved in FP32 (the full solve is what the refinement
   needs as its correction basis), X = the n vectors in FP64 and X1 = the
   maxn wanted ones.  K times

       Y = A X1,  S = X^H Y,  G = X^H X1,  R = I - G,
       lambda_j = s_jj / g_jj for the wanted j (the rest keep their FP32
       eigenvalues),
       E_ij = (s_ij + lambda_j r_ij) / (lambda_j - lambda_i) when
       |lambda_j - lambda_i| > delta and the occupations of i and j differ,
       else E_ij = r_ij / 2 (the pair is only orthonormalized),
       X1 <- X1 + X E,

   delta = 2 (max_{i != j} |s_ij| + spectral radius * max |r_ij|) of the
   first step, kept for every step: recomputed later it only grows (the
   pairs left as clusters keep their residuals), while the estimates of the
   states left as clusters are off by at most the width of their cluster.
   Pairs of equal occupation are left alone because the density matrix needs
   the occupied subspace, not the rotations inside it, and nearly degenerate
   pairs inside it are where one Newton step fails.  Afterwards every
   cluster of consecutive estimates closer than 30 delta whose occupations
   differ gets a Rayleigh-Ritz step (its H and overlap, the small generalized
   problem, the rotation): pairs with a gap of at least 30 delta start with
   mixings of at most max |s_ij| / gap, which two Newton steps take below
   1e-10, those just above delta start near 0.1 and do not.  The products
   go through the GEMMul8 bridge (15 moduli match FP64) with the full
   workspace when the device has room.

   On sidia333_nc_cluster (2n = 5616) the refined density matrix is within
   2e-11 of FP64, the SCF takes the FP64 path; the energy density matrix
   of the refined vectors is off by up to 2e-6 (the vectors inside the
   fully occupied clusters stay FP32 mixtures), which is why the solvers
   solve the step that ends the SCF in FP64. */
#include "eigen_refine_gpu.h"

#include <mpi.h>

#include <math.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuComplex.h>

/* the GEMMul8 bridge (gemmul8_bridge.cu) */
cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double *,
                                   const double *, int, const double *, int, const double *, double *, int);
cublasStatus_t openmx_gemmul8Zgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                   const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                                   const cuDoubleComplex *, cuDoubleComplex *, int);
cublasStatus_t openmx_gemmul8DgemmUnblocked(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                            const double *, const double *, int, const double *, int, const double *,
                                            double *, int);
cublasStatus_t openmx_gemmul8ZgemmUnblocked(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                            const cuDoubleComplex *, const cuDoubleComplex *, int,
                                            const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *,
                                            int);
int       openmx_gemmul8DgemmEnabled(void);
int       openmx_gemmul8ZgemmEnabled(void);
long long openmx_gemmul8NativeCalls(int is_complex);

/* LAPACK, with interleaved complex arrays */
void dsygv_(int *itype, char *jobz, char *uplo, int *n, double *a, int *lda, double *b, int *ldb, double *w,
            double *work, int *lwork, int *info);
void zhegv_(int *itype, char *jobz, char *uplo, int *n, double *a, int *lda, double *b, int *ldb, double *w,
            double *work, int *lwork, double *rwork, int *info);

static int env_flag(const char *name, int fallback)
{
    const char *value = getenv(name);

    if (value == NULL || value[0] == '\0') return fallback;
    return atoi(value) != 0;
}

static void refine_check(cudaError_t status, const char *what)
{
    if (status != cudaSuccess) {
        fprintf(stderr, "eigen_refine_gpu: %s: %s\n", what, cudaGetErrorString(status));
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void refine_check_blas(cublasStatus_t status, const char *what)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "eigen_refine_gpu: %s: cuBLAS status %d\n", what, (int)status);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

/* ------------------------------------------------------------------ */

int openmx_eigen_refine_configure(int cplx, int verbose, int *ratio)
{
    const char *value = getenv("OPENMX_EIGEN_REFINE");
    int iterations = 0;

    *ratio = 0;
    if (value != NULL && value[0] != '\0') {
        iterations = (0 < atoi(value)) ? atoi(value) : 0;
        if (verbose) {
            printf("<eigen_refine_gpu> OPENMX_EIGEN_REFINE=%s: refined FP32 eigensolver %s\n", value,
                   iterations ? "on" : "off");
            fflush(stdout);
        }
        return iterations;
    }
    {
        int device = 0, perf_ratio = 0;

        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(&perf_ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, device) == cudaSuccess &&
            8 <= perf_ratio) {
            if (cplx ? openmx_gemmul8ZgemmEnabled() : openmx_gemmul8DgemmEnabled()) {
                iterations = 2;
                *ratio = perf_ratio;
                if (verbose) {
                    printf("<eigen_refine_gpu> refined FP32 eigensolver on by default (FP32:FP64 throughput ratio %d); "
                           "OPENMX_EIGEN_REFINE=0 turns it off\n", perf_ratio);
                    fflush(stdout);
                }
            }
            else if (verbose) {
                printf("<eigen_refine_gpu> the refined FP32 eigensolver is not used: the %s GEMMul8 products are off "
                       "(FP32:FP64 throughput ratio %d)\n", cplx ? "complex" : "real", perf_ratio);
                fflush(stdout);
            }
        }
        (void)cudaGetLastError();
    }
    return iterations;
}

double openmx_eigen_refine_until(void)
{
    const char *value = getenv("OPENMX_EIGEN_REFINE_UNTIL");

    return (value != NULL && 0.0 < atof(value)) ? atof(value) : 0.0;
}

int openmx_eigen_refine_warm_enabled(void)
{
    return env_flag("OPENMX_EIGEN_REFINE_WARM", 1);
}

int openmx_eigen_refine_warm_band_enabled(void)
{
    return openmx_eigen_refine_warm_enabled() && env_flag("OPENMX_EIGEN_REFINE_WARM_BAND", 1);
}

int openmx_eigen_refine_warm_band_period(void)
{
    const char *value = getenv("OPENMX_EIGEN_REFINE_WARM_PERIOD");

    if (value == NULL || value[0] == '\0') return 4;
    return (0 < atoi(value)) ? atoi(value) : 0;
}

int openmx_eigen_refine_warm_band_step(int scf_iter)
{
    int const period = openmx_eigen_refine_warm_band_period();

    return openmx_eigen_refine_warm_band_enabled() && !(0 < period && scf_iter % period == 0);
}

static void refine_blocks_release(EigenRefineState *st)
{
    if (st->own != NULL) refine_check(cudaFree(st->own), "cudaFree blocks");
    st->own = NULL;
    st->b1 = st->b2 = NULL;
    st->products_ready = 0;
}

/* the product blocks: the FP32 scratch once the solve is done, else a
   buffer of their own; 0 when there is no room */
static int refine_blocks_ensure(EigenRefineState *st, EigenRefineDevice *dev, const EigenRefineProblem *pb,
                                EigenRefineReport *report)
{
    size_t const nk = (size_t)pb->n * (size_t)(env_flag("OPENMX_EIGEN_REFINE_ALL", 0) ? pb->n : pb->maxn);
    size_t const block_bytes = (nk * (pb->cplx ? 2 : 1) * sizeof(double) + 511u) / 512u * 512u;

    refine_blocks_release(st);
    if (2 * block_bytes <= pb->region) {
        st->b1 = (double *)pb->fp32;
    }
    else {
        if (dev->try_malloc(&st->own, 2 * block_bytes) != cudaSuccess) {
            printf("<eigen_refine_gpu> no room for the %.1f MiB of refinement products; FP64 eigensolver%s\n",
                   2.0 * (double)block_bytes / (1024.0 * 1024.0), pb->transient ? "" : " for the rest of the SCF cycle");
            fflush(stdout);
            report->persistent = pb->transient ? 0 : 1;
            return 0;
        }
        st->b1 = (double *)st->own;
    }
    st->b2 = (double *)((unsigned char *)st->b1 + block_bytes);
    return 1;
}

void openmx_eigen_refine_abandon(EigenRefineState *st)
{
    refine_blocks_release(st);
}

void openmx_eigen_refine_state_release(EigenRefineState *st)
{
    refine_blocks_release(st);
    if (st->lam != NULL) refine_check(cudaFree(st->lam), "cudaFree lam");
    if (st->occ != NULL) refine_check(cudaFree(st->occ), "cudaFree occ");
    st->lam = st->occ = NULL;
    st->capacity = 0;
    st->basis_valid = 0;
}

void openmx_eigen_refine_device_release(EigenRefineDevice *dev)
{
    if (dev->params32 != NULL) (void)cusolverDnDestroyParams(dev->params32);
    dev->params32 = NULL;
    dev->params32_tried = 0;
}

/* ------------------------------------------------------------------ */

/* The products: GEMMul8 with the full workspace when the device has room
   for it, else the blocked product; OPENMX_EIGEN_REFINE_UNBLOCKED=0 takes
   the blocked product always.  C = op(A) op(B) + beta C. */
static cublasStatus_t refine_gemm(const EigenRefineDevice *dev, int cplx, cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k, const void *A, int lda, const void *B,
                                  int ldb, double beta, void *C, int ldc)
{
    static int unblocked = -1;

    if (unblocked < 0) unblocked = env_flag("OPENMX_EIGEN_REFINE_UNBLOCKED", 1);
    if (cplx) {
        cuDoubleComplex const one = make_cuDoubleComplex(1.0, 0.0), b = make_cuDoubleComplex(beta, 0.0);

        if (unblocked)
            return openmx_gemmul8ZgemmUnblocked(dev->cublas, transa, transb, m, n, k, &one, (const cuDoubleComplex *)A,
                                                lda, (const cuDoubleComplex *)B, ldb, &b, (cuDoubleComplex *)C, ldc);
        return openmx_gemmul8Zgemm(dev->cublas, transa, transb, m, n, k, &one, (const cuDoubleComplex *)A, lda,
                                   (const cuDoubleComplex *)B, ldb, &b, (cuDoubleComplex *)C, ldc);
    }
    else {
        double const one = 1.0;
        cublasOperation_t const ta = (transa == CUBLAS_OP_C) ? CUBLAS_OP_T : transa;
        cublasOperation_t const tb = (transb == CUBLAS_OP_C) ? CUBLAS_OP_T : transb;

        if (unblocked)
            return openmx_gemmul8DgemmUnblocked(dev->cublas, ta, tb, m, n, k, &one, (const double *)A, lda,
                                                (const double *)B, ldb, &beta, (double *)C, ldc);
        return openmx_gemmul8Dgemm(dev->cublas, ta, tb, m, n, k, &one, (const double *)A, lda, (const double *)B, ldb,
                                   &beta, (double *)C, ldc);
    }
}

/* plain cuBLAS FP64, for the small products of the Rayleigh-Ritz step */
static cublasStatus_t native_gemm(const EigenRefineDevice *dev, int cplx, cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k, const void *A, int lda, const void *B,
                                  int ldb, void *C, int ldc)
{
    if (cplx) {
        cuDoubleComplex const one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);

        return cublasZgemm(dev->cublas, transa, transb, m, n, k, &one, (const cuDoubleComplex *)A, lda,
                           (const cuDoubleComplex *)B, ldb, &zero, (cuDoubleComplex *)C, ldc);
    }
    else {
        double const one = 1.0, zero = 0.0;
        cublasOperation_t const ta = (transa == CUBLAS_OP_C) ? CUBLAS_OP_T : transa;
        cublasOperation_t const tb = (transb == CUBLAS_OP_C) ? CUBLAS_OP_T : transb;

        return cublasDgemm(dev->cublas, ta, tb, m, n, k, &one, (const double *)A, lda, (const double *)B, ldb, &zero,
                           (double *)C, ldc);
    }
}

/* ------------------------------------------------------------------ */

/* The FP32 solve of all n eigenpairs of a (n x n, device) into af (the
   vectors) and wf (the eigenvalues), through the two-stage algorithm where
   cuSOLVER has it (12.3.4 and later; on an RTX 5080 323 ms instead of 638
   for n = 5616, at eigenvalue errors of 4e-6 instead of 1e-6 Ha that the
   refinement removes; OPENMX_EIGEN_FP32_TWO_STAGE=0 keeps the one-stage
   solve).  a is left untouched.  Returns 1 on success; otherwise says why
   and returns 0. */
static int refine_fp32_solve(EigenRefineDevice *dev, int cplx, int n, const void *a, float *af, float *wf)
{
    cusolverEigMode_t const jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t const  uplo = CUBLAS_FILL_MODE_LOWER;
    cudaDataType const      type = cplx ? CUDA_C_32F : CUDA_R_32F;
    size_t const            nn = (size_t)n * (size_t)n;
    size_t const            count = cplx ? 2 * nn : nn;
    const double           *ad = (const double *)a;
    float vl = 0.0f, vu = 0.0f;
    int64_t h_meig = 0;
    int32_t info = 0;
    size_t d_bytes = 0, h_bytes = 0;
    cusolverStatus_t status;
    int two_stage = 0;

#pragma acc parallel loop deviceptr(ad, af)
    for (size_t i = 0; i < count; i++) af[i] = (float)ad[i];

#if CUSOLVER_VERSION >= 12304
    {
        static int wanted = -1;

        if (wanted < 0) wanted = env_flag("OPENMX_EIGEN_FP32_TWO_STAGE", 1);
        if (wanted && !dev->params32_tried) {
            dev->params32_tried = 1;
            if (cusolverDnCreateParams(&dev->params32) == CUSOLVER_STATUS_SUCCESS &&
                cusolverDnSetAdvOptions(dev->params32, CUSOLVERDN_SYEVD, CUSOLVER_ALG_2) != CUSOLVER_STATUS_SUCCESS) {
                (void)cusolverDnDestroyParams(dev->params32);
                dev->params32 = NULL;
            }
        }
        two_stage = (wanted && dev->params32 != NULL);
    }
#endif
    {
        static int announced = 0;

        if (!announced) {
            announced = 1;
            printf("<eigen_refine_gpu> FP32 full solve: %s\n",
                   two_stage ? "two-stage cusolverDnXsyevd (CUSOLVER_ALG_2)"
                             : "one-stage cusolverDnXsyevdx (cuSOLVER before 12.3.4, or OPENMX_EIGEN_FP32_TWO_STAGE=0)");
            fflush(stdout);
        }
    }

    if (two_stage) {
#if CUSOLVER_VERSION >= 12304
        status = cusolverDnXsyevd_bufferSize(dev->cusolver, dev->params32, jobz, uplo, n, type, af, n, CUDA_R_32F, wf,
                                             type, &d_bytes, &h_bytes);
#endif
    }
    else {
        status = cusolverDnXsyevdx_bufferSize(dev->cusolver, NULL, jobz, CUSOLVER_EIG_RANGE_ALL, uplo, n, type, af, n,
                                              &vl, &vu, 1L, (int64_t)n, &h_meig, CUDA_R_32F, wf, type, &d_bytes,
                                              &h_bytes);
    }
    if (status != CUSOLVER_STATUS_SUCCESS) {
        printf("<eigen_refine_gpu> FP32 eigensolver workspace query failed (status %d); solving in FP64\n", (int)status);
        fflush(stdout);
        return 0;
    }
    if (*dev->d_work_bytes < d_bytes) {
        void *grown = NULL;

        if (dev->try_malloc(&grown, d_bytes) != cudaSuccess) {
            printf("<eigen_refine_gpu> no room for the %.1f MiB FP32 eigensolver workspace; solving in FP64\n",
                   (double)d_bytes / (1024.0 * 1024.0));
            fflush(stdout);
            return 0;
        }
        if (*dev->d_work != NULL) refine_check(cudaFree(*dev->d_work), "cudaFree workspace");
        *dev->d_work = grown;
        *dev->d_work_bytes = d_bytes;
    }
    if (*dev->h_work_bytes < h_bytes) {
        free(*dev->h_work);
        *dev->h_work = dev->host_malloc(h_bytes, 1, "cuSOLVER host workspace");
        *dev->h_work_bytes = h_bytes;
    }

    if (two_stage) {
#if CUSOLVER_VERSION >= 12304
        status = cusolverDnXsyevd(dev->cusolver, dev->params32, jobz, uplo, n, type, af, n, CUDA_R_32F, wf, type,
                                  *dev->d_work, *dev->d_work_bytes, *dev->h_work, *dev->h_work_bytes, dev->d_info);
        h_meig = n;
#endif
    }
    else {
        status = cusolverDnXsyevdx(dev->cusolver, NULL, jobz, CUSOLVER_EIG_RANGE_ALL, uplo, n, type, af, n, &vl, &vu,
                                   1L, (int64_t)n, &h_meig, CUDA_R_32F, wf, type, *dev->d_work, *dev->d_work_bytes,
                                   *dev->h_work, *dev->h_work_bytes, dev->d_info);
    }
    if (status == CUSOLVER_STATUS_SUCCESS) {
        refine_check(cudaMemcpyAsync(&info, dev->d_info, sizeof(int32_t), cudaMemcpyDeviceToHost, dev->stream),
                     "download info");
    }
    refine_check(cudaStreamSynchronize(dev->stream), "synchronize");
    if (status != CUSOLVER_STATUS_SUCCESS || info != 0 || h_meig < (int64_t)n) {
        printf("<eigen_refine_gpu> FP32 eigensolver failed (status %d, info %d, %lld of %d eigenpairs); solving in "
               "FP64\n", (int)status, (int)info, (long long)h_meig, n);
        fflush(stdout);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */

static double refine_anorm = 0.0;   /* spectral radius of the latest prepare, per process */

/* max offdiag |S| and max |I - G| of the products S (b2) and G (b1), n x kc */
static void refine_maxima(int cplx, const double *b1, const double *b2, int n, int kc, double *max_s_out,
                          double *max_r_out)
{
    int const width = cplx ? 2 : 1;
    double max_s = 0.0, max_r = 0.0;

#pragma acc parallel loop collapse(2) deviceptr(b1, b2) reduction(max : max_s, max_r)
    for (int j = 0; j < kc; j++) {
        for (int i = 0; i < n; i++) {
            size_t const at = (size_t)i + (size_t)j * (size_t)n;
            double const gr = b1[width * at] - ((i == j) ? 1.0 : 0.0);
            double const gi = cplx ? b1[width * at + 1] : 0.0;
            double const sr = b2[width * at];
            double const si = cplx ? b2[width * at + 1] : 0.0;

            if (i != j) max_s = fmax(max_s, hypot(sr, si));
            max_r = fmax(max_r, hypot(gr, gi));
        }
    }
    *max_s_out = max_s;
    *max_r_out = max_r;
}

int openmx_eigen_refine_prepare(EigenRefineState *st, EigenRefineDevice *dev, const EigenRefineProblem *pb, double *e0,
                                EigenRefineReport *report)
{
    int const cplx = pb->cplx, n = pb->n;
    size_t const nn = (size_t)n * (size_t)n;
    double *a = (double *)pb->a;
    double *x = (double *)pb->x;
    float *af = (float *)pb->fp32;
    float *wf = af + (cplx ? 2 * nn : nn);

    memset(report, 0, sizeof(*report));
    st->cplx = cplx;

    /* the caller's products that built a may still run on its streams
       (cuBLAS on a non-blocking stream is not ordered with the OpenACC
       kernels below) */
    refine_check(cudaDeviceSynchronize(), "synchronize before the refined solve");

    /* symmetric or Hermitian from the lower triangle, the part every solve
       reads: the matrix comes from products and is so only to rounding */
    if (cplx) {
#pragma acc parallel loop collapse(2) deviceptr(a)
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < n; row++) {
                size_t const at = (size_t)row + (size_t)col * (size_t)n;

                if (row < col) {
                    size_t const low = (size_t)col + (size_t)row * (size_t)n;

                    a[2 * at]     = a[2 * low];
                    a[2 * at + 1] = -a[2 * low + 1];
                }
                else if (row == col) {
                    a[2 * at + 1] = 0.0;
                }
            }
        }
    }
    else {
#pragma acc parallel loop collapse(2) deviceptr(a)
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < n; row++) {
                if (row < col) a[(size_t)row + (size_t)col * (size_t)n] = a[(size_t)col + (size_t)row * (size_t)n];
            }
        }
    }

    if (st->capacity < n) {
        openmx_eigen_refine_state_release(st);
        if (dev->try_malloc((void **)&st->lam, (size_t)n * sizeof(double)) != cudaSuccess ||
            dev->try_malloc((void **)&st->occ, (size_t)n * sizeof(double)) != cudaSuccess) {
            printf("<eigen_refine_gpu> no room for the refinement eigenvalues; FP64 eigensolver%s\n",
                   pb->transient ? "" : " for the rest of the SCF cycle");
            fflush(stdout);
            report->persistent = pb->transient ? 0 : 1;
            return 0;
        }
        st->capacity = n;
    }

    refine_blocks_release(st);
    if (pb->warm) {
        /* the first refinement step's products on the old vectors (X = X1):
           Y = A X1, S = X^H Y to b2, G = X^H X1 to b1.  Their maxima are the
           residual of the old vectors on the new matrix: above 2e-5 (an
           FP32 solve leaves 4e-6) that solve follows after all, else finish
           starts from these products */
        int const kc = env_flag("OPENMX_EIGEN_REFINE_ALL", 0) ? n : pb->maxn;   /* diagnostic: refine every column */
        long long const native0 = openmx_gemmul8NativeCalls(cplx);
        double max_s = 0.0, max_r = 0.0;
        double *b1, *b2;

        refine_check(cudaMemcpy(e0, st->lam, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost), "download lam");
        if (!refine_blocks_ensure(st, dev, pb, report)) return 0;
        b1 = st->b1;
        b2 = st->b2;
        refine_check_blas(refine_gemm(dev, cplx, CUBLAS_OP_N, CUBLAS_OP_N, n, kc, n, a, n, x, n, 0.0, b1, n), "A X1");
        refine_check_blas(refine_gemm(dev, cplx, CUBLAS_OP_C, CUBLAS_OP_N, n, kc, n, x, n, b1, n, 0.0, b2, n), "X^H Y");
        refine_check_blas(refine_gemm(dev, cplx, CUBLAS_OP_C, CUBLAS_OP_N, n, kc, n, x, n, x, n, 0.0, b1, n), "X^H X1");
        refine_check(cudaDeviceSynchronize(), "synchronize");
        if (pb->defaulted && openmx_gemmul8NativeCalls(cplx) != native0) {
            printf("<eigen_refine_gpu> the refinement products fell back to plain cuBLAS FP64 (GEMMul8 workspace "
                   "policy); FP64 eigensolver for the rest of the run\n");
            fflush(stdout);
            refine_blocks_release(st);
            st->basis_valid = 0;
            report->persistent = 2;
            return 0;
        }
        refine_maxima(cplx, b1, b2, n, kc, &max_s, &max_r);
        if (2.0e-5 < max_s || 2.0e-5 < max_r) {
            printf("<eigen_refine_gpu> warm start: residual %.1e above 2e-5; FP32 eigensolver\n", fmax(max_s, max_r));
            fflush(stdout);
            refine_blocks_release(st);
            return -1;
        }
        st->products_ready = 1;
    }
    else {
        float *w;

        st->basis_valid = 0;
        if (!refine_fp32_solve(dev, cplx, n, a, af, wf)) {
            /* a failure of the solve recurs on the same kind of matrix */
            report->persistent = pb->transient ? 0 : 1;
            return 0;
        }
        report->fp32_solved = 1;
        {
            size_t const count = cplx ? 2 * nn : nn;

#pragma acc parallel loop deviceptr(af, x)
            for (size_t i = 0; i < count; i++) x[i] = (double)af[i];
        }
        w = (float *)dev->host_malloc((size_t)n, sizeof(float), "FP32 eigenvalues");
        refine_check(cudaMemcpy(w, wf, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost), "download eigenvalues");
        for (int i = 0; i < n; i++) e0[i] = (double)w[i];
        free(w);
        refine_check(cudaMemcpy(st->lam, e0, (size_t)n * sizeof(double), cudaMemcpyHostToDevice), "upload lam");
    }
    refine_anorm = fmax(fabs(e0[0]), fabs(e0[n - 1]));
    return 1;
}

/* ------------------------------------------------------------------ */

/* Rayleigh-Ritz inside one cluster: the m columns xc (n x m, device) span an
   invariant subspace to the accuracy of the refinement but inside it they
   are the FP32 mixtures the refinement only orthonormalized.  The cluster's
   H and overlap (m x m) are formed through z (n x m) and small (2 m^2),
   each product through GEMMul8 when it is large enough for that to pay
   (about 1e9 flops, a millisecond of FP64 on a GeForce) and in cuBLAS FP64
   otherwise; the generalized problem goes to LAPACK on the host below 256
   states and to cuSOLVER on the device beyond (w_dev, m doubles, receives
   the eigenvalues there).  xc is rotated onto the eigenvectors and their
   eigenvalues go to w.  Returns 0 (xc unchanged) if the solve fails. */
static int refine_rayleigh_ritz(EigenRefineDevice *dev, int cplx, const void *a, double *xc, int n, int m, double *z,
                                double *small, double *w_dev, double *w)
{
    int const    width = cplx ? 2 : 1;
    size_t const mm = (size_t)m * (size_t)m;
    size_t const nm = (size_t)n * (size_t)m;
    int const    big = (1.0e9 <= 8.0 * (double)n * (double)n * (double)m);
    int const    mid = (1.0e9 <= 8.0 * (double)m * (double)m * (double)n);
    double      *hc = small, *nc = small + width * mm;
    int          info = 0;

    /* z = A X_c, H_c = X_c^H z, N_c = X_c^H X_c */
    refine_check_blas(big ? refine_gemm(dev, cplx, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, a, n, xc, n, 0.0, z, n)
                          : native_gemm(dev, cplx, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, a, n, xc, n, z, n),
                      "A X_c");
    refine_check_blas(mid ? refine_gemm(dev, cplx, CUBLAS_OP_C, CUBLAS_OP_N, m, m, n, xc, n, z, n, 0.0, hc, m)
                          : native_gemm(dev, cplx, CUBLAS_OP_C, CUBLAS_OP_N, m, m, n, xc, n, z, n, hc, m),
                      "X_c^H A X_c");
    refine_check_blas(mid ? refine_gemm(dev, cplx, CUBLAS_OP_C, CUBLAS_OP_N, m, m, n, xc, n, xc, n, 0.0, nc, m)
                          : native_gemm(dev, cplx, CUBLAS_OP_C, CUBLAS_OP_N, m, m, n, xc, n, xc, n, nc, m),
                      "X_c^H X_c");
    refine_check(cudaDeviceSynchronize(), "synchronize");

    if (m < 256) {
        double *h = (double *)dev->host_malloc(2 * width * mm, sizeof(double), "cluster Rayleigh-Ritz");
        double *s = h + width * mm;
        double *rwork = (double *)dev->host_malloc((size_t)(3 * m), sizeof(double), "cluster Rayleigh-Ritz");
        double query[2] = {0.0, 0.0};
        double *work;
        int itype = 1, dim = m, lwork = -1;

        refine_check(cudaMemcpy(h, small, 2 * width * mm * sizeof(double), cudaMemcpyDeviceToHost), "download cluster");
        if (cplx) zhegv_(&itype, "V", "L", &dim, h, &dim, s, &dim, w, query, &lwork, rwork, &info);
        else dsygv_(&itype, "V", "L", &dim, h, &dim, s, &dim, w, query, &lwork, &info);
        lwork = (info == 0 && 1.0 <= query[0]) ? (int)query[0] : 3 * m;
        work = (double *)dev->host_malloc((size_t)width * (size_t)lwork, sizeof(double), "cluster Rayleigh-Ritz");
        if (cplx) zhegv_(&itype, "V", "L", &dim, h, &dim, s, &dim, w, work, &lwork, rwork, &info);
        else dsygv_(&itype, "V", "L", &dim, h, &dim, s, &dim, w, work, &lwork, &info);
        free(work);
        free(rwork);
        if (info == 0) {
            /* complete before the products on the solver's stream read it */
            refine_check(cudaMemcpyAsync(small, h, width * mm * sizeof(double), cudaMemcpyHostToDevice, dev->stream),
                         "upload Y");
            refine_check(cudaStreamSynchronize(dev->stream), "synchronize");
        }
        free(h);
        if (info != 0) return 0;
    }
    else {
        cusolverStatus_t status;
        int32_t dev_info = 0;
        int lwork = 0;
        void *work = NULL;
        int own_work = 0;
        size_t need;

        if (cplx)
            status = cusolverDnZhegvd_bufferSize(dev->cusolver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
                                                 CUBLAS_FILL_MODE_LOWER, m, (cuDoubleComplex *)hc, m,
                                                 (cuDoubleComplex *)nc, m, w_dev, &lwork);
        else
            status = cusolverDnDsygvd_bufferSize(dev->cusolver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
                                                 CUBLAS_FILL_MODE_LOWER, m, hc, m, nc, m, w_dev, &lwork);
        if (status != CUSOLVER_STATUS_SUCCESS) return 0;
        need = (size_t)lwork * (size_t)width * sizeof(double);
        if (*dev->d_work_bytes < need) {
            if (dev->try_malloc(&work, need) != cudaSuccess) return 0;
            own_work = 1;
        }
        else {
            work = *dev->d_work;
        }
        if (cplx)
            status = cusolverDnZhegvd(dev->cusolver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                      m, (cuDoubleComplex *)hc, m, (cuDoubleComplex *)nc, m, w_dev,
                                      (cuDoubleComplex *)work, lwork, dev->d_info);
        else
            status = cusolverDnDsygvd(dev->cusolver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                      m, hc, m, nc, m, w_dev, (double *)work, lwork, dev->d_info);
        if (status == CUSOLVER_STATUS_SUCCESS)
            refine_check(cudaMemcpyAsync(&dev_info, dev->d_info, sizeof(int32_t), cudaMemcpyDeviceToHost, dev->stream),
                         "download info");
        refine_check(cudaStreamSynchronize(dev->stream), "synchronize");
        if (own_work) refine_check(cudaFree(work), "cudaFree");
        if (status != CUSOLVER_STATUS_SUCCESS || dev_info != 0) return 0;
        refine_check(cudaMemcpy(w, w_dev, (size_t)m * sizeof(double), cudaMemcpyDeviceToHost), "download w");
    }

    /* xc <- xc Y through z */
    refine_check_blas(mid ? refine_gemm(dev, cplx, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, xc, n, hc, m, 0.0, z, n)
                          : native_gemm(dev, cplx, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, xc, n, hc, m, z, n),
                      "X_c Y");
    refine_check(cudaDeviceSynchronize(), "synchronize");
    {
        size_t const count = width * nm;

#pragma acc parallel loop deviceptr(xc, z)
        for (size_t i = 0; i < count; i++) xc[i] = z[i];
    }
    return 1;
}

/* ------------------------------------------------------------------ */

int openmx_eigen_refine_finish(EigenRefineState *st, EigenRefineDevice *dev, const EigenRefineProblem *pb,
                               const double *f0, EigenRefineReport *report)
{
    int const    cplx = pb->cplx, n = pb->n, maxn = pb->maxn, width = cplx ? 2 : 1;
    int const    kc = env_flag("OPENMX_EIGEN_REFINE_ALL", 0) ? n : pb->maxn;   /* diagnostic: refine every column */
    size_t const nk = (size_t)n * (size_t)kc;
    double      *a = (double *)pb->a;
    double      *x = (double *)pb->x;
    double      *lam = st->lam, *occ = st->occ;
    double      *b1, *b2;
    double       max_s = 0.0, max_r = 0.0, delta = 0.0, first_delta = 0.0, chain;
    double const anorm = refine_anorm;
    long long    native0;
    int          rr_clusters = 0, rr_largest = 0, rr_ok = 1;
    /* the cluster's H and overlap (2 m^2) go into b2 (n maxn); beyond 2048
       states the cluster solve would cost more than the FP64 eigensolver
       saves on a GeForce */
    int const    rr_limit = (int)fmin(2048.0, sqrt(0.5 * (double)n * (double)kc));

    refine_check(cudaMemcpy(occ, f0, (size_t)n * sizeof(double), cudaMemcpyHostToDevice), "upload occupations");

    /* the product blocks: the FP32 scratch now that its vectors are in x
       and its eigenvalues on the host, else a buffer of their own */
    if (st->b1 == NULL && !refine_blocks_ensure(st, dev, pb, report)) return 0;
    b1 = st->b1;
    b2 = st->b2;

    native0 = openmx_gemmul8NativeCalls(cplx);
    for (int it = 0; it < pb->iterations; it++) {
        /* Y = A X1 to b1, S = X^H Y to b2, G = X^H X1 to b1 (a warm prepare
           left the first step's) */
        if (!(it == 0 && st->products_ready)) {
            refine_check_blas(refine_gemm(dev, cplx, CUBLAS_OP_N, CUBLAS_OP_N, n, kc, n, a, n, x, n, 0.0, b1, n), "A X1");
            refine_check_blas(refine_gemm(dev, cplx, CUBLAS_OP_C, CUBLAS_OP_N, n, kc, n, x, n, b1, n, 0.0, b2, n), "X^H Y");
            refine_check_blas(refine_gemm(dev, cplx, CUBLAS_OP_C, CUBLAS_OP_N, n, kc, n, x, n, x, n, 0.0, b1, n), "X^H X1");
            refine_check(cudaDeviceSynchronize(), "synchronize");
        }
        if (pb->defaulted && openmx_gemmul8NativeCalls(cplx) != native0) {
            /* plain FP64 products cost more than the FP64 eigensolver on the
               GPUs this default is for */
            printf("<eigen_refine_gpu> the refinement products fell back to plain cuBLAS FP64 (GEMMul8 workspace "
                   "policy); FP64 eigensolver for the rest of the run\n");
            fflush(stdout);
            refine_blocks_release(st);
            st->basis_valid = 0;
            report->persistent = 2;
            return 0;
        }

#pragma acc parallel loop deviceptr(b1, b2, lam)
        for (int j = 0; j < kc; j++) {
            size_t const at = (size_t)j + (size_t)j * (size_t)n;

            lam[j] = b2[width * at] / b1[width * at];
        }

        refine_maxima(cplx, b1, b2, n, kc, &max_s, &max_r);
        delta = 2.0 * (max_s + anorm * max_r);
        if (it == 0) {
            /* a warm start's residual is far below an FP32 solve's, which
               would shrink delta and the cluster chain to nothing: keep the
               FP32 solve's delta as the floor, so nearly degenerate pairs are
               treated as after a cold start (OPENMX_EIGEN_REFINE_WARM_DELTA_FLOOR=0
               drops the floor) */
            if (pb->warm) {
                if (env_flag("OPENMX_EIGEN_REFINE_WARM_DELTA_FLOOR", 1) && delta < st->delta_cold) delta = st->delta_cold;
            }
            else {
                st->delta_cold = delta;
            }
            first_delta = delta;
        }
        delta = first_delta;

        /* E in place of G */
#pragma acc parallel loop collapse(2) deviceptr(b1, b2, lam, occ)
        for (int j = 0; j < kc; j++) {
            for (int i = 0; i < n; i++) {
                size_t const at = (size_t)i + (size_t)j * (size_t)n;
                double const rr = ((i == j) ? 1.0 : 0.0) - b1[width * at];
                double const ri = cplx ? -b1[width * at + 1] : 0.0;
                double er = 0.5 * rr, ei = 0.5 * ri;

                if (i != j) {
                    double const d = lam[j] - lam[i];

                    if (delta < fabs(d) && 1.0e-12 <= fabs(occ[i] - occ[j])) {
                        er = (b2[width * at] + lam[j] * rr) / d;
                        ei = cplx ? (b2[width * at + 1] + lam[j] * ri) / d : 0.0;
                    }
                }
                b1[width * at] = er;
                if (cplx) b1[width * at + 1] = ei;
            }
        }

        /* X1 + X E, through b2 */
        {
            size_t const count = width * nk;

#pragma acc parallel loop deviceptr(b2, x)
            for (size_t i = 0; i < count; i++) b2[i] = x[i];
        }
        refine_check_blas(refine_gemm(dev, cplx, CUBLAS_OP_N, CUBLAS_OP_N, n, kc, n, x, n, b1, n, 1.0, b2, n),
                          "X1 + X E");
        refine_check(cudaDeviceSynchronize(), "synchronize");
        {
            size_t const count = width * nk;

#pragma acc parallel loop deviceptr(b2, x)
            for (size_t i = 0; i < count; i++) x[i] = b2[i];
        }
    }

    /* Rayleigh-Ritz inside every cluster of consecutive estimates closer
       than chain whose occupations differ (for Fermi occupations the same
       as holding a partially occupied state, and also true of an orbital
       emptied by index) */
    chain = 30.0 * first_delta;
    {
        double *lh = (double *)dev->host_malloc((size_t)kc, sizeof(double), "refined eigenvalues");
        double *fh = (double *)dev->host_malloc((size_t)kc, sizeof(double), "refined occupations");
        int *c0s = (int *)dev->host_malloc((size_t)kc, sizeof(int), "refinement clusters");
        int *ms = (int *)dev->host_malloc((size_t)kc, sizeof(int), "refinement clusters");
        int clusters = 0;

        refine_check(cudaMemcpy(lh, lam, (size_t)kc * sizeof(double), cudaMemcpyDeviceToHost), "download lam");
        for (int j = 0; j < kc; j++) fh[j] = pb->occupation(lh[j], j + 1, pb->occupation_ctx);
        for (int c0 = 0; c0 < kc;) {
            int c1 = c0;
            double f_low = fh[c0], f_high = fh[c0];

            while (c1 + 1 < kc && fabs(lh[c1 + 1] - lh[c1]) <= chain) c1++;
            for (int j = c0; j <= c1; j++) {
                f_low = fmin(f_low, fh[j]);
                f_high = fmax(f_high, fh[j]);
            }
            if (c0 < c1 && 1.0e-12 <= f_high - f_low) {
                c0s[clusters] = c0;
                ms[clusters] = c1 - c0 + 1;
                if (rr_largest < ms[clusters]) rr_largest = ms[clusters];
                clusters++;
            }
            c0 = c1 + 1;
        }
        if (rr_limit < rr_largest) {
            printf("<eigen_refine_gpu> a partially occupied cluster of %d states (limit %d) is too large for the "
                   "refined FP32 eigensolver; FP64 eigensolver for the rest of the SCF cycle\n", rr_largest, rr_limit);
            fflush(stdout);
            rr_ok = 0;
        }
        for (int c = 0; rr_ok && c < clusters; c++) {
            /* occ is free by now: it serves as the device eigenvalue buffer */
            if (refine_rayleigh_ritz(dev, cplx, a, x + (size_t)width * (size_t)c0s[c] * (size_t)n, n, ms[c], b1, b2,
                                     occ, lh + c0s[c])) {
                rr_clusters++;
            }
            else {
                printf("<eigen_refine_gpu> the Rayleigh-Ritz step of a cluster of %d states failed; FP64 eigensolver "
                       "for the rest of the SCF cycle\n", ms[c]);
                fflush(stdout);
                rr_ok = 0;
            }
        }
        if (rr_ok) refine_check(cudaMemcpy(lam, lh, (size_t)kc * sizeof(double), cudaMemcpyHostToDevice), "upload lam");
        free(lh);
        free(fh);
        free(c0s);
        free(ms);
    }
    if (!rr_ok) {
        /* a still holds the symmetric matrix: the caller solves it in FP64 */
        refine_blocks_release(st);
        st->basis_valid = 0;
        report->persistent = 1;
        return 0;
    }

    {
        size_t const count = (size_t)width * (size_t)n * (size_t)maxn;
        double *w = pb->w;

#pragma acc parallel loop deviceptr(a, x)
        for (size_t i = 0; i < count; i++) a[i] = x[i];
#pragma acc parallel loop deviceptr(w, lam)
        for (int i = 0; i < maxn; i++) w[i] = lam[i];
    }

    refine_blocks_release(st);
    st->basis_valid = 1;
    st->basis_n = n;

    report->columns = kc;
    report->max_r = max_r;
    report->max_s = max_s;
    report->delta = first_delta;
    report->chain = chain;
    report->rr_clusters = rr_clusters;
    report->rr_largest = rr_largest;
    return 1;
}
