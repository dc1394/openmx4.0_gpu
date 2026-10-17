#include "openmx_stdout_level.h"
/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include "openmx_common.h"
#include "hip_runtime_compat.h"
#include "hipsolver_compat.h"
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* magma_openmx.cu */
int    openmx_gpu_zheevd_phase_in(void *d_A, int n, hipStream_t stream);
int    openmx_gpu_eigen_check(int cplx, int phase, void *d_V, int n, int cols, const double *d_w,
                              int *d_bad, hipStream_t stream);
int    openmx_gpu_eigen_check_input(int cplx, const void *d_A, int n, int *d_bad, hipStream_t stream);
int    openmx_gpusolver_use_evd(int n, int maxn);
int    openmx_gpu_eigen_retry_policy(void);
size_t openmx_gpusolver_cache_limit(void);
int    openmx_magma_eigen_gpu(int cplx, int n, int maxn, void *d_A, double *w, int *mout);

void openmx_gpusolver_cache_release(void);

/*
  The generic dense GPU eigensolver wrappers: the lowest MaxN eigenpairs of
  the m x m matrix A (column major, lower triangle), the eigenvalues in
  W[0..MaxN-1] and the eigenvectors in the first MaxN columns of A; the rest
  of A and of W is left as it was.  Users: Eigen_lapack and EigenBand_lapack
  (m >= GPU_CPU_SWITCH_NUM), the DC non-collinear solve (the _openmp complex
  entry) and openmx_gpusolver_dense_utils.h (DMmu, CWF, Cluster LNO, electric
  polarization), which consume only those MaxN columns and eigenvalues.

  Port of the NVIDIA build's cached wrappers (25d2eda0): every entry point
  shares one context per host thread (handle, stream, device scratch and the
  workspace query), so the many small DC, LNO and Wannier solves no longer
  create a handle, a stream and every allocation per call.  Scratch beyond
  OPENMX_GPUSOLVER_CACHE_MB (alias OPENMX_CUSOLVER_CACHE_MB, default 64 MiB)
  is freed after each solve, all of it inside an OpenMP parallel region, and
  openmx_gpusolver_cache_release() frees the calling thread's context.

  rocSOLVER solves a complete spectrum with its divide and conquer driver
  ({sy,he}evd, much faster than the subset driver), the complex one through
  the phase similarity of magma_openmx.cu, and a partial one with
  {sy,he}evdx (see openmx_gpusolver_use_evd).  The eigenvalues pass an
  internal n-entry array (rocSOLVER uses all n entries of W).  A failed
  rocSOLVER solve (nonzero info, an error, or a non-finite eigenvalue or
  vector element) is solved again with MAGMA from the caller's unchanged
  input (openmx_gpu_eigen_retry_policy).  An input whose lower triangle has
  a non-finite element is refused before solving.

  Return value: 0, the solver's positive info, -1 for invalid arguments,
  -2 for non-finite eigenpairs, -1000-status for a HIP or hipSOLVER error, or
  MAGMA's negative error of the retry.  Nothing is written to A or W unless
  0 is returned; nothing aborts, and no call is repeated (the old
  wait_hipfunc retried a failed solve on its already overwritten matrix).

  The host entries take host arrays; the _openmp entries take device arrays,
  host arrays, or host arrays mapped by OpenMP, which stand for their device
  copies (the NVIDIA OpenACC entries' present/host_data use_device).  Every
  transfer has the context's HIP device memory on one side, so that HIP
  copies it on the GPU for every kind of pointer.
*/

typedef struct {
    int                 initialized, device;
    int                 query_n, query_maxn, query_kind, query_lwork;
    hipsolverDnHandle_t handle;
    hipStream_t         stream;
    void               *a, *work;
    double             *w, *host_w;
    int                *info;        /* device: [0] info, [1] non-finite flag */
    size_t              a_bytes, w_bytes, work_bytes, host_w_bytes;
} GpusolverContext;

static _Thread_local GpusolverContext solver_ctx;

#define GPUSOLVER_NONFINITE (-2)

static int solver_error(const char *call, int status)
{
    fprintf(stderr, "OpenMX GPU eigensolver: %s failed (status=%d).\n", call, status);
    fflush(stderr);
    return -1000 - status;
}

#define SOLVER_CHECK(call) do { int status_ = (int)(call);                                  \
    if (status_ != 0) { (void)hipGetLastError(); return solver_error(#call, status_); } } while (0)

void openmx_gpusolver_cache_release(void)
{
    GpusolverContext *c = &solver_ctx;
    int current, device;

    if (!c->initialized) return;
    if (hipGetDevice(&current) != hipSuccess) {
        (void)hipGetLastError();
        return;
    }
    device = c->device;
    if (current != device && hipSetDevice(device) != hipSuccess) {
        (void)hipGetLastError();
        return;
    }
    (void)hipStreamSynchronize(c->stream);
    if (c->a) (void)hipFree(c->a);
    if (c->w) (void)hipFree(c->w);
    if (c->work) (void)hipFree(c->work);
    if (c->info) (void)hipFree(c->info);
    free(c->host_w);
    (void)hipsolverDnDestroy(c->handle);
    (void)hipStreamDestroy(c->stream);
    memset(c, 0, sizeof(*c));
    if (current != device) (void)hipSetDevice(current);
    (void)hipGetLastError();
}

static int solver_init(void)
{
    GpusolverContext *c = &solver_ctx;
    GpusolverContext next;
    int device, status;

    memset(&next, 0, sizeof(next));
    SOLVER_CHECK(hipGetDevice(&device));
    if (c->initialized && c->device == device) return 0;
    if (c->initialized) openmx_gpusolver_cache_release();
    status = (int)hipsolverDnCreate(&next.handle);
    if (status) goto fail;
    status = (int)hipStreamCreateWithFlags(&next.stream, hipStreamNonBlocking);
    if (status) goto fail;
    status = (int)hipsolverDnSetStream(next.handle, next.stream);
    if (status) goto fail;
    status = (int)hipMalloc((void **)&next.info, 2 * sizeof(int));
    if (status) goto fail;
    next.device = device;
    next.initialized = 1;
    *c = next;
    return 0;
fail:
    (void)hipGetLastError();
    if (next.info) (void)hipFree(next.info);
    if (next.handle) (void)hipsolverDnDestroy(next.handle);
    if (next.stream) (void)hipStreamDestroy(next.stream);
    return solver_error("context initialization", status);
}

/* grow-only device scratch */
static int solver_reserve(void **p, size_t *capacity, size_t bytes)
{
    if (bytes <= *capacity) return 0;
    if (*p) SOLVER_CHECK(hipFree(*p));
    *p = NULL;
    *capacity = 0;
    SOLVER_CHECK(hipMalloc(p, bytes));
    *capacity = bytes;
    return 0;
}

/* Large solves must not keep another dense matrix next to the Band and
   Cluster resident arenas; small ones keep their allocations, within a
   bounded per-thread footprint.  The handle, the stream and info stay. */
static void solver_trim(void)
{
    GpusolverContext *c = &solver_ctx;

    /* Wannier functions call the wrappers from OpenMP worker threads, whose
       contexts no serial suite boundary could release */
    if (omp_in_parallel()) {
        openmx_gpusolver_cache_release();
        return;
    }
    if (c->a_bytes + c->w_bytes + c->work_bytes + c->host_w_bytes <= openmx_gpusolver_cache_limit()) return;
    if (c->a) (void)hipFree(c->a);
    if (c->w) (void)hipFree(c->w);
    if (c->work) (void)hipFree(c->work);
    free(c->host_w);
    c->a = c->work = NULL;
    c->w = c->host_w = NULL;
    c->a_bytes = c->w_bytes = c->work_bytes = c->host_w_bytes = 0;
    c->query_n = c->query_maxn = c->query_kind = c->query_lwork = 0;
}

/* an OpenMP-mapped host array stands for its device copy; any other
   pointer is used as it is */
static void *solver_device_view(void *p)
{
    int   device = omp_get_default_device();
    void *mapped;

    if (p == NULL || device == omp_get_initial_device() || !omp_target_is_present(p, device)) return p;
    mapped = omp_get_mapped_ptr(p, device);
    return (mapped != NULL) ? mapped : p;
}

/* rocSOLVER on the staged matrix c->a: the lowest maxn eigenpairs, the
   eigenvalues in c->w and the vectors in the first maxn columns of c->a.
   0, the solver's positive info, GPUSOLVER_NONFINITE or -1000-status. */
static int solver_run(GpusolverContext *c, int n, int maxn, int cplx)
{
    const int                evd = openmx_gpusolver_use_evd(n, maxn);
    const int                kind = 2 * cplx + (evd ? 0 : 1);  /* 0 dsyevd, 1 dsyevdx, 2 zheevd, 3 zheevdx */
    const size_t             width = cplx ? sizeof(hipDoubleComplex) : sizeof(double);
    const hipsolverEigMode_t jobz = HIPSOLVER_EIG_MODE_VECTOR;
    const hipblasFillMode_t  uplo = HIPBLAS_FILL_MODE_LOWER;
    int                      h_info[2] = {0, 0};
    int                      nev = maxn;
    int                      result;

    if (c->query_n != n || c->query_maxn != maxn || c->query_kind != kind) {
        int lwork = 0;

        switch (kind) {
        case 0:
            SOLVER_CHECK(hipsolverDnDsyevd_bufferSize(c->handle, jobz, uplo, n, (const double *)c->a, n, c->w,
                                                      &lwork));
            break;
        case 1:
            SOLVER_CHECK(hipsolverDnDsyevdx_bufferSize(c->handle, jobz, HIPSOLVER_EIG_RANGE_I, uplo, n,
                                                       (const double *)c->a, n, 0.0, 0.0, 1, maxn, &nev, c->w,
                                                       &lwork));
            break;
        case 2:
            SOLVER_CHECK(hipsolverDnZheevd_bufferSize(c->handle, jobz, uplo, n, (const hipDoubleComplex *)c->a, n,
                                                      c->w, &lwork));
            break;
        default:
            SOLVER_CHECK(hipsolverDnZheevdx_bufferSize(c->handle, jobz, HIPSOLVER_EIG_RANGE_I, uplo, n,
                                                       (const hipDoubleComplex *)c->a, n, 0.0, 0.0, 1, maxn, &nev,
                                                       c->w, &lwork));
            break;
        }
        if (lwork < 1) lwork = 1;
        c->query_n = n;
        c->query_maxn = maxn;
        c->query_kind = kind;
        c->query_lwork = lwork;
    }
    result = solver_reserve(&c->work, &c->work_bytes, width * (size_t)c->query_lwork);
    if (result) return result;

    if (cplx && evd) SOLVER_CHECK(openmx_gpu_zheevd_phase_in(c->a, n, c->stream));
    switch (kind) {
    case 0:
        SOLVER_CHECK(hipsolverDnDsyevd(c->handle, jobz, uplo, n, (double *)c->a, n, c->w, (double *)c->work,
                                       c->query_lwork, c->info));
        break;
    case 1:
        SOLVER_CHECK(hipsolverDnDsyevdx(c->handle, jobz, HIPSOLVER_EIG_RANGE_I, uplo, n, (double *)c->a, n,
                                        0.0, 0.0, 1, maxn, &nev, c->w, (double *)c->work, c->query_lwork,
                                        c->info));
        break;
    case 2:
        SOLVER_CHECK(hipsolverDnZheevd(c->handle, jobz, uplo, n, (hipDoubleComplex *)c->a, n, c->w,
                                       (hipDoubleComplex *)c->work, c->query_lwork, c->info));
        break;
    default:
        SOLVER_CHECK(hipsolverDnZheevdx(c->handle, jobz, HIPSOLVER_EIG_RANGE_I, uplo, n,
                                        (hipDoubleComplex *)c->a, n, 0.0, 0.0, 1, maxn, &nev, c->w,
                                        (hipDoubleComplex *)c->work, c->query_lwork, c->info));
        break;
    }
    /* c->info[1]: a non-finite value among the pairs returned (the complex
       divide and conquer also completes its phase similarity here) */
    SOLVER_CHECK(openmx_gpu_eigen_check(cplx, cplx && evd, c->a, n, maxn, c->w, c->info + 1, c->stream));
    SOLVER_CHECK(hipMemcpyAsync(h_info, c->info, sizeof(h_info), hipMemcpyDeviceToHost, c->stream));
    SOLVER_CHECK(hipStreamSynchronize(c->stream));
    if (h_info[0] != 0) return h_info[0];
    /* the subset driver may add eigenvalues degenerate with the last one;
       the one of ROCm 7.2 also returns none at all (info 0) for the
       identity, the zero matrix or a diagonal with repeated entries, which
       the MAGMA retry then solves */
    if (!evd && nev < maxn) return solver_error("eigenpair count", nev);
    if (h_info[1] != 0) return GPUSOLVER_NONFINITE;
    return 0;
}

/* a failed solve that MAGMA does not solve either: the input's state */
static void solver_report_input(GpusolverContext *c, const void *src, int n, int cplx, int result)
{
    const size_t count = (size_t)n * (size_t)n * (cplx ? 2u : 1u);
    double      *h = (double *)malloc(count * sizeof(double));
    size_t       nonfinite = 0, i;
    double       amax = 0.0, asym = 0.0;
    int          r, col;

    if (h == NULL ||
        hipMemcpyAsync(c->a, src, count * sizeof(double), hipMemcpyDefault, c->stream) != hipSuccess ||
        hipMemcpyAsync(h, c->a, count * sizeof(double), hipMemcpyDeviceToHost, c->stream) != hipSuccess ||
        hipStreamSynchronize(c->stream) != hipSuccess) {
        (void)hipGetLastError();
        free(h);
        fprintf(stderr, "OpenMX GPU eigensolver: n=%d failed (%d); the input could not be examined.\n", n, result);
        fflush(stderr);
        return;
    }
    for (i = 0; i < count; i++) {
        if (!isfinite(h[i])) nonfinite++;
        else if (fabs(h[i]) > amax) amax = fabs(h[i]);
    }
    for (col = 0; col < n; col++) {
        for (r = col + 1; r < n; r++) {
            const size_t lo = (size_t)col * n + r, up = (size_t)r * n + col;
            const double d = cplx ? fabs(h[2 * lo] - h[2 * up]) + fabs(h[2 * lo + 1] + h[2 * up + 1])
                                  : fabs(h[lo] - h[up]);

            if (d > asym) asym = d;
        }
    }
    fprintf(stderr, "OpenMX GPU eigensolver: n=%d failed (%d): the input has %zu non-finite values, "
            "max |a| = %.3e, max |a_ij - conj(a_ji)| = %.3e (both triangles; the solvers read the lower)\n",
            n, result, nonfinite, amax, asym);
    fflush(stderr);
    free(h);
}

/* the failed (or, in the retry test, the successful) rocSOLVER solve of
   c->a again with MAGMA, from the unchanged input src; the eigenvalues end
   up in c->w as after rocSOLVER */
static int solver_retry_magma(GpusolverContext *c, const void *src, int n, int maxn, int cplx, int first,
                              int forced)
{
    static _Thread_local int announced = 0;
    const size_t abytes = (size_t)n * (size_t)n * (cplx ? sizeof(hipDoubleComplex) : sizeof(double));
    int          mout = 0;
    int          result;

    if (forced) {
        if (!announced) {
            announced = 1;
            fprintf(stderr, "OpenMX GPU eigensolver: retry test (OPENMX_GPUSOLVER_RETRY_TEST=1): every rocSOLVER "
                    "solve is solved again with MAGMA (first n=%d).\n", n);
            fflush(stderr);
        }
    }
    else {
        if (OPENMX_STDOUT_LEVEL(2)) {
            fprintf(stderr, "OpenMX GPU eigensolver: rocSOLVER %s failed (%d, n=%d, %d pairs); solving again with "
                    "MAGMA.\n", cplx ? "complex" : "real", first, n, maxn);
            fflush(stderr);
        }
    }

    /* MAGMA works on its own queue */
    SOLVER_CHECK(hipStreamSynchronize(c->stream));
    SOLVER_CHECK(hipMemcpyAsync(c->a, src, abytes, hipMemcpyDefault, c->stream));
    SOLVER_CHECK(hipStreamSynchronize(c->stream));
    if (c->host_w_bytes < sizeof(double) * (size_t)n) {
        void *p = realloc(c->host_w, sizeof(double) * (size_t)n);

        if (p == NULL) return solver_error("host eigenvalue buffer allocation", 1);
        c->host_w = (double *)p;
        c->host_w_bytes = sizeof(double) * (size_t)n;
    }

    result = openmx_magma_eigen_gpu(cplx, n, maxn, c->a, c->host_w, &mout);
    if (result == 0 && mout < maxn) result = solver_error("MAGMA eigenpair count", mout);
    if (result == 0) {
        int bad = 0;

        /* the eigenvalues back into c->w; MAGMA's eigenpairs are checked as
           rocSOLVER's were */
        SOLVER_CHECK(hipMemcpyAsync(c->w, c->host_w, sizeof(double) * (size_t)maxn, hipMemcpyHostToDevice,
                                    c->stream));
        SOLVER_CHECK(openmx_gpu_eigen_check(cplx, 0, c->a, n, maxn, c->w, c->info + 1, c->stream));
        SOLVER_CHECK(hipMemcpyAsync(&bad, c->info + 1, sizeof(int), hipMemcpyDeviceToHost, c->stream));
        SOLVER_CHECK(hipStreamSynchronize(c->stream));
        if (bad) result = GPUSOLVER_NONFINITE;
    }
    if (result == 0) {
        if (!forced) {
            if (OPENMX_STDOUT_LEVEL(2)) {
                fprintf(stderr, "OpenMX GPU eigensolver: the MAGMA solve succeeded (n=%d).\n", n);
                fflush(stderr);
            }
        }
    }
    else {
        solver_report_input(c, src, n, cplx, result);
    }
    return result;
}

static int solver_solve(void *A, double *W, int n, int maxn, int cplx, int mapped)
{
    GpusolverContext *c = &solver_ctx;
    const size_t      width = cplx ? sizeof(hipDoubleComplex) : sizeof(double);
    void             *src = A;
    double           *dst_w = W;
    size_t            abytes;
    int               policy, result;

    if (n < 0 || maxn < 0 || maxn > n) return -1;
    if (n == 0 || maxn == 0) return 0;
    if (A == NULL || W == NULL) return -1;
    abytes = width * (size_t)n * (size_t)n;
    if (mapped) {
        src = solver_device_view(A);
        dst_w = (double *)solver_device_view(W);
    }

    result = solver_init();
    if (result) return result;
    result = solver_reserve(&c->a, &c->a_bytes, abytes);
    if (!result) result = solver_reserve((void **)&c->w, &c->w_bytes, sizeof(double) * (size_t)n);
    if (!result) {
        result = (int)hipMemcpyAsync(c->a, src, abytes, hipMemcpyDefault, c->stream);
        if (result) {
            (void)hipGetLastError();
            result = solver_error("matrix transfer", result);
        }
    }
    if (result) {
        openmx_gpusolver_cache_release();
        return result;
    }

    /* a NaN can come back as finite eigenpairs of another matrix */
    {
        int bad = 0;

        result = openmx_gpu_eigen_check_input(cplx, c->a, n, c->info + 1, c->stream);
        if (!result) result = (int)hipMemcpyAsync(&bad, c->info + 1, sizeof(int), hipMemcpyDeviceToHost, c->stream);
        if (!result) result = (int)hipStreamSynchronize(c->stream);
        if (result) {
            (void)hipGetLastError();
            result = solver_error("input check", result);
        }
        else if (bad) {
            result = GPUSOLVER_NONFINITE;
            solver_report_input(c, src, n, cplx, result);
        }
        if (result) {
            openmx_gpusolver_cache_release();
            return result;
        }
    }

    policy = openmx_gpu_eigen_retry_policy();
    result = solver_run(c, n, maxn, cplx);
    if (policy != 0 && (result != 0 || policy == 2)) {
        result = solver_retry_magma(c, src, n, maxn, cplx, result, result == 0);
    }

    /* only the requested eigenpairs are part of the contract */
    if (result == 0) {
        int status = (int)hipMemcpyAsync(dst_w, c->w, sizeof(double) * (size_t)maxn, hipMemcpyDefault, c->stream);

        if (!status) status = (int)hipMemcpyAsync(src, c->a, width * (size_t)n * (size_t)maxn, hipMemcpyDefault,
                                                  c->stream);
        if (!status) status = (int)hipStreamSynchronize(c->stream);
        if (status) {
            (void)hipGetLastError();
            result = solver_error("result transfer", status);
        }
    }
    if (result) openmx_gpusolver_cache_release();
    else solver_trim();
    return result;
}

int32_t gpusolver_Syevdx(double * A, double * W, int32_t m, int32_t MaxN)
{
    return solver_solve(A, W, m, MaxN, 0, 0);
}

int32_t gpusolver_Syevdx_openmp(double * A, double * W, int32_t m, int32_t MaxN)
{
    return solver_solve(A, W, m, MaxN, 0, 1);
}

int32_t gpusolver_Syevdx_Complex(dcomplex * A, double * W, int32_t m, int32_t MaxN)
{
    return solver_solve(A, W, m, MaxN, 1, 0);
}

int32_t gpusolver_Syevdx_Complex_openmp(dcomplex * A, double * W, int32_t m, int32_t MaxN)
{
    return solver_solve(A, W, m, MaxN, 1, 1);
}
