/* cuSOLVER policy shared by the Band, Cluster, DC, LNO and Krylov paths.
 * CUDA 13.4 Update 1 (cuSOLVER 12.3.4) adds SYEVD algorithm 2 and an
 * architecture/math-mode/problem-size dependent selector in algorithm 0.
 * Default to that selector for complete spectra; preserve SYEVDX's subset
 * computation for partial spectra. Explicit settings retain manual control.
 * FP64
 * emulation is enabled on the solver handle: cuBLAS environment variables
 * alone do not configure cuSOLVER's private cuBLAS handles. CUDA 13.4 cuBLAS
 * chooses Ozaki-II when it is faster than Ozaki-I/native FP64.
 * https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
 * https://docs.nvidia.com/cuda/cusolver/#floating-point-emulation
 */
#define OPENMX_CUSOLVER_IMPLEMENTATION
#include "openmx_cusolver_compat.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

typedef struct {
    int initialized;
    int algorithm;                 /* -1: legacy, 0: automatic, 1/2: stages */
    int minimum_set;
    int64_t two_stage_min_n;
    cusolverDnParams_t params;
    cusolverDnParams_t one_stage_params;
} OpenMXCuSolverPolicy;

/* A params object is never shared by concurrent host threads. Handles and
   their workspaces remain owned by their existing OpenMX contexts. */
static _Thread_local OpenMXCuSolverPolicy policy;

static int openmx_solver_verbose(void)
{
    const char *s = getenv("OPENMX_CUSOLVER_VERBOSE");
    return s && atoi(s) != 0;
}

static cusolverStatus_t openmx_solver_policy_init(void)
{
    if (policy.initialized) return CUSOLVER_STATUS_SUCCESS;
    policy.algorithm = -1;
    {
        const char *s = getenv("OPENMX_CUSOLVER_TWO_STAGE_MIN_N");
        if (s && *s) {
            char *end;
            long long value;
            errno = 0;
            value = strtoll(s, &end, 10);
            if (!errno && !*end && value >= 0) {
                policy.two_stage_min_n = value;
                policy.minimum_set = 1;
            }
            else fprintf(stderr, "OpenMX: invalid OPENMX_CUSOLVER_TWO_STAGE_MIN_N=%s; ignoring override.\n", s);
        }
    }
#if CUSOLVER_VERSION >= 12304
    {
        int major = 0, minor = 0, patch = 0;
        const char *s = getenv("OPENMX_CUSOLVER_ALGORITHM");
        cusolverStatus_t st;
        st = cusolverGetProperty(MAJOR_VERSION, &major);
        if (st != CUSOLVER_STATUS_SUCCESS) return st;
        st = cusolverGetProperty(MINOR_VERSION, &minor);
        if (st != CUSOLVER_STATUS_SUCCESS) return st;
        st = cusolverGetProperty(PATCH_LEVEL, &patch);
        if (st != CUSOLVER_STATUS_SUCCESS) return st;
        if (major * 1000 + minor * 100 + patch >= 12304) {
            /* A valid standalone minimum keeps the previous manual-threshold
               interface. With neither override, cuSOLVER chooses the crossover
               instead of hard-coding a dimension measured on one GPU. */
            policy.algorithm = policy.minimum_set ? 2 : 0;
            if (s && (!strcmp(s, "legacy") || !strcmp(s, "0")))
                policy.algorithm = -1;
            else if (s && !strcmp(s, "auto")) policy.algorithm = 0;
            else if (s && (!strcmp(s, "one-stage") || !strcmp(s, "1")))
                policy.algorithm = 1;
            else if (s && (!strcmp(s, "two-stage") || !strcmp(s, "2")))
                policy.algorithm = 2;
            else if (s && *s) {
                policy.algorithm = 0;
                fprintf(stderr, "OpenMX: invalid OPENMX_CUSOLVER_ALGORITHM=%s; using auto.\n", s);
            }
            if (policy.algorithm >= 0) {
                st = cusolverDnCreateParams(&policy.params);
                if (st != CUSOLVER_STATUS_SUCCESS) return st;
                st = cusolverDnSetAdvOptions(policy.params, CUSOLVERDN_SYEVD,
                                             (cusolverAlgMode_t)policy.algorithm);
                if (st != CUSOLVER_STATUS_SUCCESS) {
                    (void)cusolverDnDestroyParams(policy.params);
                    policy.params = NULL;
                    policy.algorithm = -1;
                    if (st != CUSOLVER_STATUS_NOT_SUPPORTED && st != CUSOLVER_STATUS_INVALID_VALUE)
                        return st;
                }
                if (policy.algorithm == 2 && policy.two_stage_min_n > 0) {
                    st = cusolverDnCreateParams(&policy.one_stage_params);
                    if (st == CUSOLVER_STATUS_SUCCESS)
                        st = cusolverDnSetAdvOptions(policy.one_stage_params,
                                                     CUSOLVERDN_SYEVD, CUSOLVER_ALG_1);
                    if (st != CUSOLVER_STATUS_SUCCESS) {
                        if (policy.one_stage_params)
                            (void)cusolverDnDestroyParams(policy.one_stage_params);
                        (void)cusolverDnDestroyParams(policy.params);
                        policy.one_stage_params = policy.params = NULL;
                        return st;
                    }
                }
            }
        }
        if (openmx_solver_verbose()) {
            if (policy.algorithm == 0)
                fprintf(stderr, "OpenMX: cuSOLVER %d.%d.%d, eigen algorithm=auto, "
                        "crossover=cuSOLVER automatic (size/architecture/math mode).\n",
                        major, minor, patch);
            else if (policy.algorithm == 2)
                fprintf(stderr, "OpenMX: cuSOLVER %d.%d.%d, eigen algorithm=two-stage, "
                        "two-stage minimum n=%lld.\n", major, minor, patch,
                        (long long)policy.two_stage_min_n);
            else
                fprintf(stderr, "OpenMX: cuSOLVER %d.%d.%d, eigen algorithm=%s.\n",
                        major, minor, patch, policy.algorithm == 1 ? "one-stage" : "legacy");
        }
    }
#endif
    policy.initialized = 1;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t openmx_cusolverDnCreate(cusolverDnHandle_t *handle)
{
    cusolverStatus_t st = cusolverDnCreate(handle);
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
#if CUSOLVER_VERSION >= 12200
    {
        const char *s = getenv("OPENMX_CUSOLVER_EMULATION");
        if (!s || atoi(s) != 0) {
            st = cusolverDnSetMathMode(*handle, CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH);
            if (st == CUSOLVER_STATUS_SUCCESS)
                st = cusolverDnSetEmulationStrategy(*handle, CUDA_EMULATION_STRATEGY_PERFORMANT);
            if (st == CUSOLVER_STATUS_SUCCESS)
                st = cusolverDnSetFixedPointEmulationMantissaControl(
                    *handle, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC);
            if (st == CUSOLVER_STATUS_NOT_SUPPORTED || st == CUSOLVER_STATUS_INVALID_VALUE) {
                st = cusolverDnSetMathMode(*handle, CUSOLVER_DEFAULT_MATH);
                if (openmx_solver_verbose())
                    fprintf(stderr, "OpenMX: cuSOLVER FP64 emulation unavailable; using native FP64.\n");
            }
            if (st != CUSOLVER_STATUS_SUCCESS) {
                (void)cusolverDnDestroy(*handle);
                *handle = NULL;
                return st;
            }
        }
    }
#endif
    return CUSOLVER_STATUS_SUCCESS;
}

static int openmx_solver_type_bit(cudaDataType type)
{
    return type == CUDA_R_64F ? 1 : type == CUDA_C_64F ? 2 : 0;
}

static int openmx_solver_use_full(cusolverDnParams_t params, cudaDataType type,
                                  cudaDataType wtype, cudaDataType compute, int64_t n)
{
    int bit = openmx_solver_type_bit(type);
    return !params && policy.algorithm >= 0 && bit &&
           (policy.algorithm != 2 || n >= policy.two_stage_min_n) &&
           wtype == CUDA_R_64F && compute == type;
}

static cusolverDnParams_t openmx_solver_full_params(cusolverDnParams_t params,
    cudaDataType type, cudaDataType wtype, cudaDataType compute, int64_t n)
{
    if (openmx_solver_use_full(params, type, wtype, compute, n)) return policy.params;
    /* A manual minimum must also hold for callers of the full-spectrum API:
       passing NULL here would permit ALG_0 to choose two-stage below the floor. */
    if (!params && policy.one_stage_params && n < policy.two_stage_min_n &&
        openmx_solver_type_bit(type) && wtype == CUDA_R_64F && compute == type)
        return policy.one_stage_params;
    return params;
}

/* FP64 emulation retry.  The emulated fixed-point eigensolver of a GeForce
   fails to converge now and then (info > 0: RTX 5080, sidia333, band Col
   info=395 and band NonCol info=126, once in a few hundred solves) and the
   callers abort.  A solve in an FP64-emulated handle is therefore made from
   a copy of A (on the device, or in host memory when the device is short),
   and info > 0 is answered by restoring A and solving once more in native
   FP64; the handle returns to emulation afterwards.  The workspace queries
   return the larger of the two modes' requirements so the retry fits. */
static int openmx_solver_emulated(cusolverDnHandle_t handle, cudaDataType type)
{
#if CUSOLVER_VERSION >= 12200
    cusolverMathMode_t mode;
    if (type != CUDA_C_64F && type != CUDA_R_64F) return 0;
    if (cusolverDnGetMathMode(handle, &mode) != CUSOLVER_STATUS_SUCCESS) return 0;
    return mode == CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH;
#else
    (void)handle; (void)type;
    return 0;
#endif
}

/* OPENMX_CUSOLVER_RETRY_TEST=1 exercises the retry on every emulated solve */
static int openmx_solver_retry_test(void)
{
    static int state = -1;
    if (state < 0) {
        const char *v = getenv("OPENMX_CUSOLVER_RETRY_TEST");
        state = (v != NULL && v[0] != '\0' && atoi(v) != 0) ? 1 : 0;
    }
    return state;
}

/* when even the native solve fails: what did the input look like? */
static void openmx_solver_report_matrix(const void *copy, int copy_host, cudaDataType type,
                                        int64_t lda, int64_t n, int h_info)
{
    const size_t width = (type == CUDA_C_64F) ? sizeof(cuDoubleComplex) : sizeof(double);
    const size_t abytes = width * (size_t)lda * (size_t)n;
    const double *h = NULL;
    double *tmp = NULL;
    size_t nan_count = 0, i;
    double amax = 0.0, herm = 0.0;
    int64_t r, c;

    if (copy_host) h = (const double *)copy;
    else {
        tmp = (double *)malloc(abytes);
        if (tmp == NULL || cudaMemcpy(tmp, copy, abytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            free(tmp);
            fprintf(stderr, "OpenMX: native FP64 solve failed too (info=%d); the input could not be examined.\n", h_info);
            return;
        }
        h = tmp;
    }
    for (i = 0; i < abytes / sizeof(double); i++) {
        if (h[i] != h[i] || h[i] - h[i] != 0.0) nan_count++;
        else if (fabs(h[i]) > amax) amax = fabs(h[i]);
    }
    if (type == CUDA_C_64F) {
        for (c = 0; c < n; c++)
            for (r = 0; r < n; r++) {
                const double re = h[2 * ((size_t)c * lda + r)], im = h[2 * ((size_t)c * lda + r) + 1];
                const double re2 = h[2 * ((size_t)r * lda + c)], im2 = h[2 * ((size_t)r * lda + c) + 1];
                const double d = fabs(re - re2) + fabs(im + im2);
                if (d > herm) herm = d;
            }
    }
    fprintf(stderr, "OpenMX: native FP64 solve failed too (info=%d, n=%lld): input has %zu non-finite values, "
            "max |a| = %.3e, max |a_ij - conj(a_ji)| = %.3e\n", h_info, (long long)n, nan_count, amax, herm);
    free(tmp);
}

typedef cusolverStatus_t (*openmx_solver_once_fn)(void *ctx);

/* runs fn once; in an emulated handle, again in native FP64 when info > 0 */
static cusolverStatus_t openmx_solver_retry(cusolverDnHandle_t handle, cudaDataType type,
                                            void *A, int64_t lda, int64_t n, int *info,
                                            openmx_solver_once_fn fn, void *ctx)
{
    cusolverStatus_t st;
#if CUSOLVER_VERSION >= 12200
    void *copy = NULL;
    int copy_host = 0;
    cudaStream_t stream = 0;
    size_t abytes = 0;

    if (n > 0 && lda > 0 && info != NULL && openmx_solver_emulated(handle, type)) {
        size_t const width = type == CUDA_C_64F ? sizeof(cuDoubleComplex) : sizeof(double);
        if ((uint64_t)lda > SIZE_MAX / width ||
            (uint64_t)n > SIZE_MAX / (width * (size_t)lda))
            return CUSOLVER_STATUS_INVALID_VALUE;
        abytes = width * (size_t)lda * (size_t)n;
        st = cusolverDnGetStream(handle, &stream);
        if (st != CUSOLVER_STATUS_SUCCESS) return st;
        if (cudaMalloc(&copy, abytes) == cudaSuccess) {
            if (cudaMemcpyAsync(copy, A, abytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                (void)cudaFree(copy);
                return CUSOLVER_STATUS_EXECUTION_FAILED;
            }
        }
        else {
            (void)cudaGetLastError();
            copy = malloc(abytes);
            copy_host = 1;
            if (copy != NULL && (cudaStreamSynchronize(stream) != cudaSuccess ||
                cudaMemcpy(copy, A, abytes, cudaMemcpyDeviceToHost) != cudaSuccess)) {
                free(copy);
                return CUSOLVER_STATUS_EXECUTION_FAILED;
            }
        }
    }
    st = fn(ctx);
    if (copy != NULL) {
        if (st == CUSOLVER_STATUS_SUCCESS) {
            int h_info = 0;
            const int forced = openmx_solver_retry_test();
            if (cudaStreamSynchronize(stream) != cudaSuccess ||
                cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
                st = CUSOLVER_STATUS_EXECUTION_FAILED;
            }
            else if (h_info > 0 || (h_info == 0 && forced)) {
                cusolverStatus_t restore_status;
                fprintf(stderr, "OpenMX: cuSOLVER FP64 emulation %s (info=%d, n=%lld); "
                        "solving again in native FP64.\n",
                        (h_info > 0) ? "did not converge" : "retry test", h_info, (long long)n);
                fflush(stderr);
                if (cudaMemcpyAsync(A, copy, abytes,
                                    copy_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice,
                                    stream) != cudaSuccess) {
                    st = CUSOLVER_STATUS_EXECUTION_FAILED;
                }
                else {
                    st = cusolverDnSetMathMode(handle, CUSOLVER_DEFAULT_MATH);
                    if (st == CUSOLVER_STATUS_SUCCESS) {
                        st = fn(ctx);
                        /* Complete the solve and any host-backed restore before
                           freeing its input or changing the handle's mode. */
                        if (cudaStreamSynchronize(stream) != cudaSuccess)
                            st = CUSOLVER_STATUS_EXECUTION_FAILED;
                        if (st == CUSOLVER_STATUS_SUCCESS &&
                            cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
                            st = CUSOLVER_STATUS_EXECUTION_FAILED;
                        restore_status = cusolverDnSetMathMode(handle, CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH);
                        if (st == CUSOLVER_STATUS_SUCCESS) st = restore_status;
                        if (st == CUSOLVER_STATUS_SUCCESS) {
                            if (h_info != 0) openmx_solver_report_matrix(copy, copy_host, type, lda, n, h_info);
                            else fprintf(stderr, "OpenMX: the native FP64 solve succeeded (n=%lld).\n", (long long)n);
                        }
                    }
                    else {
                        /* The restore may still be queued even when changing
                           the math mode failed; do not free a host copy yet. */
                        if (cudaStreamSynchronize(stream) != cudaSuccess)
                            st = CUSOLVER_STATUS_EXECUTION_FAILED;
                    }
                }
                if (st != CUSOLVER_STATUS_SUCCESS)
                    fprintf(stderr, "OpenMX: native FP64 retry failed (status %d).\n", (int)st);
                fflush(stderr);
            }
        }
        if (copy_host) free(copy);
        else if (cudaFree(copy) != cudaSuccess && st == CUSOLVER_STATUS_SUCCESS)
            st = CUSOLVER_STATUS_EXECUTION_FAILED;
    }
#else
    (void)handle; (void)type; (void)A; (void)lda; (void)n; (void)info;
    st = fn(ctx);
#endif
    return st;
}

/* the larger of the emulated and native workspace requirements */
static cusolverStatus_t openmx_solver_both_modes(cusolverDnHandle_t handle, cudaDataType type,
                                     openmx_solver_once_fn fn, void *ctx,
                                     size_t *dbytes, size_t *hbytes)
{
#if CUSOLVER_VERSION >= 12200
    size_t d0 = *dbytes, h0 = *hbytes;
    cusolverStatus_t st, restore_status;
    if (!openmx_solver_emulated(handle, type)) return CUSOLVER_STATUS_SUCCESS;
    st = cusolverDnSetMathMode(handle, CUSOLVER_DEFAULT_MATH);
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
    st = fn(ctx);
    if (st == CUSOLVER_STATUS_SUCCESS) {
        if (*dbytes < d0) *dbytes = d0;
        if (*hbytes < h0) *hbytes = h0;
    }
    else {
        *dbytes = d0;
        *hbytes = h0;
    }
    restore_status = cusolverDnSetMathMode(handle, CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH);
    return st == CUSOLVER_STATUS_SUCCESS ? restore_status : st;
#else
    (void)handle; (void)type; (void)fn; (void)ctx; (void)dbytes; (void)hbytes;
    return CUSOLVER_STATUS_SUCCESS;
#endif
}

typedef struct {
    cusolverDnHandle_t handle; cusolverDnParams_t params; cusolverEigMode_t jobz;
    cublasFillMode_t uplo; int64_t n; cudaDataType type; const void *A; int64_t lda;
    cudaDataType wtype; const void *W; cudaDataType compute; size_t *dbytes; size_t *hbytes;
    void *dwork; size_t dbytes_v; void *hwork; size_t hbytes_v; int *info;
} openmx_syevd_ctx;

typedef struct {
    cusolverDnHandle_t handle; cusolverDnParams_t params; cusolverEigMode_t jobz;
    cusolverEigRange_t range; cublasFillMode_t uplo; int64_t n; cudaDataType type;
    const void *A; int64_t lda; void *vl; void *vu; int64_t il; int64_t iu; int64_t *meig;
    cudaDataType wtype; const void *W; cudaDataType compute; size_t *dbytes; size_t *hbytes;
    void *dwork; size_t dbytes_v; void *hwork; size_t hbytes_v; int *info;
} openmx_syevdx_ctx;

static cusolverStatus_t openmx_syevd_bufferSize_once(void *vctx)
{
    openmx_syevd_ctx *c = (openmx_syevd_ctx *)vctx;
    return cusolverDnXsyevd_bufferSize(c->handle,
            openmx_solver_full_params(c->params, c->type, c->wtype, c->compute, c->n),
            c->jobz, c->uplo, c->n, c->type, c->A, c->lda, c->wtype, c->W, c->compute,
            c->dbytes, c->hbytes);
}

static cusolverStatus_t openmx_syevd_once(void *vctx)
{
    openmx_syevd_ctx *c = (openmx_syevd_ctx *)vctx;
    return cusolverDnXsyevd(c->handle,
        openmx_solver_full_params(c->params, c->type, c->wtype, c->compute, c->n),
        c->jobz, c->uplo, c->n, c->type, (void *)c->A, c->lda, c->wtype, (void *)c->W, c->compute,
        c->dwork, c->dbytes_v, c->hwork, c->hbytes_v, c->info);
}

cusolverStatus_t openmx_cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType type, const void *A, int64_t lda,
    cudaDataType wtype, const void *W, cudaDataType compute, size_t *dbytes, size_t *hbytes)
{
    cusolverStatus_t st = openmx_solver_policy_init();
    openmx_syevd_ctx c = {handle, params, jobz, uplo, n, type, A, lda, wtype, W, compute,
                          dbytes, hbytes, NULL, 0, NULL, 0, NULL};
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
    st = openmx_syevd_bufferSize_once(&c);
    if (st == CUSOLVER_STATUS_SUCCESS)
        st = openmx_solver_both_modes(handle, type, openmx_syevd_bufferSize_once, &c, dbytes, hbytes);
    return st;
}

cusolverStatus_t openmx_cusolverDnXsyevd(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType type, void *A, int64_t lda,
    cudaDataType wtype, void *W, cudaDataType compute, void *dwork, size_t dbytes,
    void *hwork, size_t hbytes, int *info)
{
    cusolverStatus_t st = openmx_solver_policy_init();
    openmx_syevd_ctx c = {handle, params, jobz, uplo, n, type, A, lda, wtype, W, compute,
                          NULL, NULL, dwork, dbytes, hwork, hbytes, info};
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
    return openmx_solver_retry(handle, type, A, lda, n, info, openmx_syevd_once, &c);
}

/* SYEVDX has no two-stage algorithm. In automatic mode, only complete-spectrum
   requests go through SYEVD's architecture/size/math-mode based selector.
   Partial requests retain SYEVDX so that automatic dispatch never expands a
   small requested subset to a full solve. Explicit stage modes may request
   that expansion and expose only the requested prefix. W may have only iu elements;
   reserve a private n-element W in the queried device workspace. Alignment
   of the solver's following workspace is preserved. */
static size_t openmx_solver_wbytes(int64_t n)
{
    return ((size_t)n * sizeof(double) + 255) & ~(size_t)255;
}

static int openmx_solver_supported_range(cusolverEigRange_t range, int64_t n,
                                         int64_t il, int64_t iu)
{
    if (n <= 0) return 0;
    if (range == CUSOLVER_EIG_RANGE_ALL) return 1;
    if (range != CUSOLVER_EIG_RANGE_I || il != 1 || iu < il || iu > n) return 0;
    return policy.algorithm != 0 || iu == n;
}

static cusolverStatus_t openmx_syevdx_bufferSize_once(void *vctx)
{
    openmx_syevdx_ctx *c = (openmx_syevdx_ctx *)vctx;
    cusolverDnHandle_t handle = c->handle; cusolverDnParams_t params = c->params;
    cusolverEigMode_t jobz = c->jobz; cusolverEigRange_t range = c->range; cublasFillMode_t uplo = c->uplo;
    int64_t n = c->n; cudaDataType type = c->type; const void *A = c->A; int64_t lda = c->lda;
    void *vl = c->vl; void *vu = c->vu; int64_t il = c->il; int64_t iu = c->iu; int64_t *meig = c->meig;
    cudaDataType wtype = c->wtype; const void *W = c->W; cudaDataType compute = c->compute;
    size_t *dbytes = c->dbytes; size_t *hbytes = c->hbytes;
    cusolverStatus_t st;
    if (openmx_solver_supported_range(range, n, il, iu) &&
        openmx_solver_use_full(params, type, wtype, compute, n)) {
        st = cusolverDnXsyevd_bufferSize(handle, policy.params, jobz, uplo, n,
                                          type, A, lda, wtype, W, compute, dbytes, hbytes);
        if (st == CUSOLVER_STATUS_SUCCESS) {
            if (*dbytes > SIZE_MAX - openmx_solver_wbytes(n)) return CUSOLVER_STATUS_INVALID_VALUE;
            *dbytes += openmx_solver_wbytes(n);
            return st;
        }
        return st;
    }
    return cusolverDnXsyevdx_bufferSize(handle, params, jobz, range, uplo, n,
        type, A, lda, vl, vu, il, iu, meig, wtype, W, compute, dbytes, hbytes);
}

cusolverStatus_t openmx_cusolverDnXsyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n, cudaDataType type,
    const void *A, int64_t lda, void *vl, void *vu, int64_t il, int64_t iu,
    int64_t *meig, cudaDataType wtype, const void *W, cudaDataType compute,
    size_t *dbytes, size_t *hbytes)
{
    cusolverStatus_t st = openmx_solver_policy_init();
    openmx_syevdx_ctx c = {handle, params, jobz, range, uplo, n, type, A, lda, vl, vu, il, iu, meig,
                           wtype, W, compute, dbytes, hbytes, NULL, 0, NULL, 0, NULL};
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
    st = openmx_syevdx_bufferSize_once(&c);
    if (st == CUSOLVER_STATUS_SUCCESS)
        st = openmx_solver_both_modes(handle, type, openmx_syevdx_bufferSize_once, &c, dbytes, hbytes);
    return st;
}

static cusolverStatus_t openmx_syevdx_once(void *vctx)
{
    openmx_syevdx_ctx *c = (openmx_syevdx_ctx *)vctx;
    cusolverDnHandle_t handle = c->handle; cusolverDnParams_t params = c->params;
    cusolverEigMode_t jobz = c->jobz; cusolverEigRange_t range = c->range; cublasFillMode_t uplo = c->uplo;
    int64_t n = c->n; cudaDataType type = c->type; void *A = (void *)c->A; int64_t lda = c->lda;
    void *vl = c->vl; void *vu = c->vu; int64_t il = c->il; int64_t iu = c->iu; int64_t *meig = c->meig;
    cudaDataType wtype = c->wtype; void *W = (void *)c->W; cudaDataType compute = c->compute;
    void *dwork = c->dwork; size_t dbytes = c->dbytes_v; void *hwork = c->hwork; size_t hbytes = c->hbytes_v;
    int *info = c->info;
    cusolverStatus_t st;
    if (openmx_solver_supported_range(range, n, il, iu) &&
        openmx_solver_use_full(params, type, wtype, compute, n)) {
        size_t extra = openmx_solver_wbytes(n);
        cudaStream_t stream;
        int64_t count = range == CUSOLVER_EIG_RANGE_ALL ? n : iu;
        if (!dwork || dbytes < extra || !meig) return CUSOLVER_STATUS_INVALID_VALUE;
        st = cusolverDnXsyevd(handle, policy.params, jobz, uplo, n, type, A, lda,
                              wtype, dwork, compute, (char *)dwork + extra,
                              dbytes - extra, hwork, hbytes, info);
        if (st != CUSOLVER_STATUS_SUCCESS) return st;
        st = cusolverDnGetStream(handle, &stream);
        if (st != CUSOLVER_STATUS_SUCCESS) return st;
        if (cudaMemcpyAsync(W, dwork, (size_t)count * sizeof(double),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return CUSOLVER_STATUS_EXECUTION_FAILED;
        *meig = count;
        return CUSOLVER_STATUS_SUCCESS;
    }
    return cusolverDnXsyevdx(handle, params, jobz, range, uplo, n, type, A, lda,
                            vl, vu, il, iu, meig, wtype, W, compute,
                            dwork, dbytes, hwork, hbytes, info);
}

cusolverStatus_t openmx_cusolverDnXsyevdx(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n, cudaDataType type,
    void *A, int64_t lda, void *vl, void *vu, int64_t il, int64_t iu,
    int64_t *meig, cudaDataType wtype, void *W, cudaDataType compute,
    void *dwork, size_t dbytes, void *hwork, size_t hbytes, int *info)
{
    cusolverStatus_t st = openmx_solver_policy_init();
    openmx_syevdx_ctx c = {handle, params, jobz, range, uplo, n, type, A, lda, vl, vu, il, iu, meig,
                           wtype, W, compute, NULL, NULL, dwork, dbytes, hwork, hbytes, info};
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
    return openmx_solver_retry(handle, type, A, lda, n, info, openmx_syevdx_once, &c);
}
