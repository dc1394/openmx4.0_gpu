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
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <openacc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Share the cache between all wrapper entry points. The larger Band/Cluster/DC
   paths maintain their own resident contexts, but local dense and LNO solves
   used to create a handle, stream and every allocation for each atom/SCF step. */
typedef struct {
    int initialized, device, query_n, query_maxn;
    cudaDataType query_type;
    cusolverDnHandle_t handle;
    cudaStream_t stream;
    void *a, *work, *host;
    double *w;
    int *info;
    size_t a_bytes, w_bytes, work_bytes, host_bytes;
    size_t query_device_bytes, query_host_bytes;
} GpusolverContext;

static _Thread_local GpusolverContext solver_ctx;

static int solver_error(const char *call, int status)
{
    fprintf(stderr, "OpenMX GPU eigensolver: %s failed (status=%d).\n", call, status);
    return -1000-status;
}

#define SOLVER_CHECK(call) do { int status_ = (int)(call); \
    if (status_ != 0) return solver_error(#call, status_); } while (0)

void openmx_gpusolver_cache_release(void)
{
    GpusolverContext *c = &solver_ctx;
    int current;
    if (!c->initialized) return;
    if (cudaGetDevice(&current) != cudaSuccess) return;
    if (cudaSetDevice(c->device) != cudaSuccess) return;
    (void)cudaStreamSynchronize(c->stream);
    if (c->a) (void)cudaFree(c->a);
    if (c->w) (void)cudaFree(c->w);
    if (c->work) (void)cudaFree(c->work);
    if (c->info) (void)cudaFree(c->info);
    free(c->host);
    (void)cusolverDnDestroy(c->handle);
    (void)cudaStreamDestroy(c->stream);
    memset(c, 0, sizeof(*c));
    (void)cudaSetDevice(current);
}

static int solver_init(void)
{
    GpusolverContext *c = &solver_ctx;
    GpusolverContext next = {0};
    int device, status;
    SOLVER_CHECK(cudaGetDevice(&device));
    if (c->initialized && c->device == device) return 0;
    if (c->initialized) openmx_gpusolver_cache_release();
    status = (int)cusolverDnCreate(&next.handle);
    if (status) goto fail;
    status = (int)cudaStreamCreateWithFlags(&next.stream, cudaStreamNonBlocking);
    if (status) goto fail;
    status = (int)cusolverDnSetStream(next.handle, next.stream);
    if (status) goto fail;
    status = (int)cudaMalloc((void **)&next.info, sizeof(int));
    if (status) goto fail;
    next.device = device;
    next.initialized = 1;
    *c = next;
    return 0;
fail:
    if (next.info) (void)cudaFree(next.info);
    if (next.handle) (void)cusolverDnDestroy(next.handle);
    if (next.stream) (void)cudaStreamDestroy(next.stream);
    return solver_error("context initialization", status);
}

static int solver_reserve(void **p, size_t *capacity, size_t bytes)
{
    if (bytes <= *capacity) return 0;
    if (*p) SOLVER_CHECK(cudaFree(*p));
    *p = NULL;
    *capacity = 0;
    SOLVER_CHECK(cudaMalloc(p, bytes));
    *capacity = bytes;
    return 0;
}

static void solver_trim(void)
{
    GpusolverContext *c = &solver_ctx;
    size_t limit = 64u * 1024u * 1024u;
    const char *s = getenv("OPENMX_CUSOLVER_CACHE_MB");
#ifdef _OPENMP
    /* Wannier calls the wrappers from worker threads. A serial test-suite
       boundary cannot release those thread-local contexts. */
    if (omp_in_parallel()) {
        openmx_gpusolver_cache_release();
        return;
    }
#endif
    if (s && *s && *s != '-') {
        char *end;
        unsigned long long mb;
        errno = 0;
        mb = strtoull(s, &end, 10);
        if (!errno && !*end && mb <= SIZE_MAX / (1024u * 1024u))
            limit = (size_t)mb * 1024u * 1024u;
    }
    /* Large host solves must not retain another dense matrix next to the
       Band/Cluster resident arenas or force-stage buffers. Keep small solves'
       allocation savings within a bounded per-rank footprint. */
    if (c->a_bytes + c->w_bytes + c->work_bytes + c->host_bytes <= limit) return;
    if (c->a) (void)cudaFree(c->a);
    if (c->w) (void)cudaFree(c->w);
    if (c->work) (void)cudaFree(c->work);
    free(c->host);
    c->a = c->work = c->host = NULL;
    c->w = NULL;
    c->a_bytes = c->w_bytes = c->work_bytes = c->host_bytes = 0;
    c->query_n = c->query_maxn = 0;
}

/* d_A is always device resident. Keep an n-element eigenvalue array even
   when the caller's OpenACC W mapping contains only MaxN entries. */
static int solver_run(void *d_A, double *W, int n, int maxn,
                      cudaDataType type, int host_output, void *host_A)
{
    GpusolverContext *c = &solver_ctx;
    int result, info = 0;
    int64_t meig = 0;
    double vl = 0.0, vu = 0.0;
    cusolverEigRange_t range = n == maxn ? CUSOLVER_EIG_RANGE_ALL : CUSOLVER_EIG_RANGE_I;
    result = solver_reserve((void **)&c->w, &c->w_bytes, sizeof(double)*(size_t)n);
    if (result) return result;
    if (c->query_n != n || c->query_maxn != maxn || c->query_type != type) {
        SOLVER_CHECK(cusolverDnXsyevdx_bufferSize(c->handle, NULL,
            CUSOLVER_EIG_MODE_VECTOR, range, CUBLAS_FILL_MODE_LOWER, n, type,
            d_A, n, &vl, &vu, 1, maxn, &meig, CUDA_R_64F, c->w, type,
            &c->query_device_bytes, &c->query_host_bytes));
        c->query_n = n;
        c->query_maxn = maxn;
        c->query_type = type;
    }
    result = solver_reserve(&c->work, &c->work_bytes, c->query_device_bytes);
    if (result) return result;
    if (c->query_host_bytes > c->host_bytes) {
        void *p = realloc(c->host, c->query_host_bytes);
        if (!p) return solver_error("host workspace allocation", 1);
        c->host = p;
        c->host_bytes = c->query_host_bytes;
    }
    SOLVER_CHECK(cusolverDnXsyevdx(c->handle, NULL, CUSOLVER_EIG_MODE_VECTOR,
        range, CUBLAS_FILL_MODE_LOWER, n, type, d_A, n, &vl, &vu, 1, maxn,
        &meig, CUDA_R_64F, c->w, type, c->work, c->query_device_bytes,
        c->host, c->query_host_bytes, c->info));
    SOLVER_CHECK(cudaMemcpyAsync(W, c->w, sizeof(double)*(size_t)maxn,
        host_output ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice, c->stream));
    if (host_A) {
        size_t width = type == CUDA_C_64F ? sizeof(cuDoubleComplex) : sizeof(double);
        SOLVER_CHECK(cudaMemcpyAsync(host_A, d_A, width*(size_t)n*(size_t)maxn,
                                     cudaMemcpyDeviceToHost, c->stream));
    }
    SOLVER_CHECK(cudaMemcpyAsync(&info, c->info, sizeof(info), cudaMemcpyDeviceToHost, c->stream));
    SOLVER_CHECK(cudaStreamSynchronize(c->stream));
    if (info == 0 && meig != maxn) return solver_error("eigenpair count", (int)meig);
    return info;
}

static int solver_host(void *A, double *W, int n, int maxn, cudaDataType type)
{
    GpusolverContext *c = &solver_ctx;
    size_t width = type == CUDA_C_64F ? sizeof(cuDoubleComplex) : sizeof(double);
    int result;
    if (n < 0 || maxn < 0 || maxn > n) return -1;
    if (n == 0 || maxn == 0) return 0;
    result = solver_init();
    if (result) return result;
    result = solver_reserve(&c->a, &c->a_bytes, width*(size_t)n*(size_t)n);
    if (result) {
        openmx_gpusolver_cache_release();
        return result;
    }
    result = (int)cudaMemcpyAsync(c->a, A, width*(size_t)n*(size_t)n,
                                  cudaMemcpyHostToDevice, c->stream);
    if (result) {
        openmx_gpusolver_cache_release();
        return solver_error("matrix transfer", result);
    }
    /* Only the requested eigenvector columns are part of the contract. */
    result = solver_run(c->a, W, n, maxn, type, 1, A);
    if (result) openmx_gpusolver_cache_release();
    else solver_trim();
    return result;
}

int32_t gpusolver_Syevdx(double *A, double *W, int32_t m, int32_t MaxN)
{
    return solver_host(A, W, m, MaxN, CUDA_R_64F);
}

int32_t gpusolver_Syevdx_Complex(dcomplex *A, double *W, int32_t m, int32_t MaxN)
{
    return solver_host(A, W, m, MaxN, CUDA_C_64F);
}

int32_t gpusolver_Syevdx_openacc(double *A, double *W, int32_t m, int32_t MaxN)
{
    int result;
    if (m < 0 || MaxN < 0 || MaxN > m) return -1;
    if (m == 0 || MaxN == 0) return 0;
    result = solver_init();
    if (result) return result;
    /* Establish dependency on any outstanding OpenACC producer before the
       independent nonblocking cuSOLVER stream accesses its data. */
    acc_wait_all();
#pragma acc data present(A[0:(size_t)m*m], W[0:MaxN])
#pragma acc host_data use_device(A, W)
    {
        result = solver_run(A, W, m, MaxN, CUDA_R_64F, 0, NULL);
    }
    if (result) openmx_gpusolver_cache_release();
    else solver_trim();
    return result;
}

int32_t gpusolver_Syevdx_Complex_openacc(dcomplex *A, double *W, int32_t m, int32_t MaxN)
{
    int result;
    if (m < 0 || MaxN < 0 || MaxN > m) return -1;
    if (m == 0 || MaxN == 0) return 0;
    result = solver_init();
    if (result) return result;
    acc_wait_all();
#pragma acc data present(A[0:(size_t)m*m], W[0:MaxN])
#pragma acc host_data use_device(A, W)
    {
        result = solver_run(A, W, m, MaxN, CUDA_C_64F, 0, NULL);
    }
    if (result) openmx_gpusolver_cache_release();
    else solver_trim();
    return result;
}

int32_t gpusolver_Syevdx_Complex_openacc_cached(dcomplex *A, double *W, int32_t m, int32_t MaxN)
{
    return gpusolver_Syevdx_Complex_openacc(A, W, m, MaxN);
}
