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

cusolverStatus_t openmx_cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType type, const void *A, int64_t lda,
    cudaDataType wtype, const void *W, cudaDataType compute, size_t *dbytes, size_t *hbytes)
{
    cusolverStatus_t st = openmx_solver_policy_init();
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
    st = cusolverDnXsyevd_bufferSize(handle,
            openmx_solver_full_params(params, type, wtype, compute, n),
            jobz, uplo, n, type, A, lda, wtype, W, compute, dbytes, hbytes);
    return st;
}

cusolverStatus_t openmx_cusolverDnXsyevd(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType type, void *A, int64_t lda,
    cudaDataType wtype, void *W, cudaDataType compute, void *dwork, size_t dbytes,
    void *hwork, size_t hbytes, int *info)
{
    cusolverStatus_t st = openmx_solver_policy_init();
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
    return cusolverDnXsyevd(handle,
        openmx_solver_full_params(params, type, wtype, compute, n),
        jobz, uplo, n, type, A, lda, wtype, W, compute, dwork, dbytes, hwork, hbytes, info);
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

cusolverStatus_t openmx_cusolverDnXsyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n, cudaDataType type,
    const void *A, int64_t lda, void *vl, void *vu, int64_t il, int64_t iu,
    int64_t *meig, cudaDataType wtype, const void *W, cudaDataType compute,
    size_t *dbytes, size_t *hbytes)
{
    cusolverStatus_t st = openmx_solver_policy_init();
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
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

cusolverStatus_t openmx_cusolverDnXsyevdx(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n, cudaDataType type,
    void *A, int64_t lda, void *vl, void *vu, int64_t il, int64_t iu,
    int64_t *meig, cudaDataType wtype, void *W, cudaDataType compute,
    void *dwork, size_t dbytes, void *hwork, size_t hbytes, int *info)
{
    cusolverStatus_t st = openmx_solver_policy_init();
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
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
