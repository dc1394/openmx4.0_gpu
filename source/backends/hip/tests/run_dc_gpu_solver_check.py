#!/usr/bin/env python3
"""Check the GPU solves of the divide-conquer method (source/Divide_Conquer.c).

The production helpers are extracted verbatim and built against
  default: host stubs of HIP/hipBLAS (a bounds-checked "device" heap, geam
           and GEMM on the CPU), or
  --gpu:   the real HIP runtime and hipBLAS (C prototypes of the compat
           headers; needs a GPU, a few MB of device memory),
with the MAGMA/rocSOLVER eigensolvers replaced by LAPACK on the host.

Non-collinear (DCNonCol_GpuSolve): for real, magnetic, spin-orbit-like and
block-diagonal (degenerate) spinor Hamiltonians the GPU solve must give the
eigenvalues and the density/energy-density matrices sum_s f(e_s) c_s c_s^H
of the CPU path of DC_NonCol (whose loops are reproduced here), the
generalized residual H C = diag(S,S) C E must vanish, and injected solver
failures (INFO > 0, MAGMA_ERR_DEVICE_ALLOC, short eigenpair count, NaN
vectors or values, a device allocation failure) must return 0 with S and H
untouched.  The superseded row-major hand-off of the old OpenMP path is
evaluated too: it diagonalized the conjugate matrix, which flips the sign of
the imaginary density-matrix parts of a magnetic cluster.

Collinear (DCCol_*): the row staging with the device transpose, the S-cache
store and the resident-overlap solve must reproduce the previous host-packed
staging bit for bit (C, eigenvalues, overlap eigenvectors), across varying
cluster sizes (the bounds-checked heap catches a buffer smaller than the
cluster it is used for).
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def function(source, name):
    match = re.search(r"^static [^;{}]*\b" + re.escape(name) + r"\([^;]*?\)\s*\{",
                      source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production function {name}")
    brace = source.index("{", match.start())
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def public(source, name):
    match = re.search(r"^void " + re.escape(name) + r"\(void\)\s*\{", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production function {name}")
    brace = source.index("{", match.start())
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def define(source, name):
    match = re.search(r"^#define\s+" + re.escape(name) + r"\b.*$", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production macro {name}")
    return match.group(0)


def typedef(source, name):
    end = source.index("} " + name + ";") + len("} " + name + ";")
    start = source.rfind("typedef struct {", 0, end)
    return source[start:end]


def prototype(source, name):
    match = re.search(r"^hipblasStatus_t " + re.escape(name) + r"\([^;]*\);", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production prototype {name}")
    return match.group(0)


COMMON = r'''
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); \
} } while (0)

typedef int MPI_Comm;
static MPI_Comm mpi_comm_level1 = 0;
static int MPI_Comm_rank(MPI_Comm comm, int *rank) { (void)comm; *rank = 0; return 0; }
static int level_stdout = 0;
static int omp_in_parallel(void) { return 0; }
static void DC_AbortWithMessage(const char *message)
{
    fprintf(stderr, "Unexpected abort: %s\n", message);
    exit(3);
}
#define wait_hipfunc(call) CHECK((int)(call) == 0)

extern void dgemm_(const char *, const char *, const int *, const int *, const int *, const double *,
                   const double *, const int *, const double *, const int *, const double *, double *,
                   const int *);
extern void dsyevd_(const char *, const char *, const int *, double *, const int *, double *, double *,
                    const int *, int *, const int *, int *);
extern void zheevd_(const char *, const char *, const int *, void *, const int *, double *, void *,
                    const int *, double *, const int *, int *, const int *, int *);
'''

CPU_DEVICE = r'''
typedef struct { double r, i; } dcomplex;
typedef struct { double x, y; } hipDoubleComplex;
#define make_hipDoubleComplex(r, i) ((hipDoubleComplex){(double)(r), (double)(i)})
typedef int hipError_t;
typedef struct ihipStream_t *hipStream_t;
typedef struct hipblasContext *hipblasHandle_t;
typedef int hipblasStatus_t;
typedef enum { HIPBLAS_OP_N = 111, HIPBLAS_OP_T = 112, HIPBLAS_OP_C = 113 } hipblasOperation_t;
enum { hipSuccess = 0, hipErrorOutOfMemory = 2, hipStreamNonBlocking = 1, hipMemcpyHostToHost = 0,
       hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2, hipMemcpyDeviceToDevice = 3 };
enum { HIPBLAS_STATUS_SUCCESS = 0, HIPBLAS_STATUS_NOT_INITIALIZED = 1, HIPBLAS_STATUS_ALLOC_FAILED = 2,
       HIPBLAS_STATUS_INVALID_VALUE = 3, HIPBLAS_STATUS_ARCH_MISMATCH = 4, HIPBLAS_STATUS_MAPPING_ERROR = 5,
       HIPBLAS_STATUS_EXECUTION_FAILED = 6, HIPBLAS_STATUS_INTERNAL_ERROR = 7,
       HIPBLAS_STATUS_NOT_SUPPORTED = 8 };

/* the "device": a registry of live allocations; every device operand of a
   copy or a BLAS call must lie inside one of them */
#define MAX_ALLOCS 512
static struct { char *base; size_t bytes; } device_heap[MAX_ALLOCS];
static int device_live, pinned_live, fail_malloc_countdown = -1;
static void device_check(const void *p, size_t bytes, int line)
{
    const char *q = (const char *)p;
    for (int i = 0; i < MAX_ALLOCS; i++) {
        if (device_heap[i].base != NULL && device_heap[i].base <= q &&
            q + bytes <= device_heap[i].base + device_heap[i].bytes) return;
    }
    fprintf(stderr, "FAIL: device range %p+%zu outside every allocation (stub line %d)\n", p, bytes, line);
    exit(2);
}
static hipError_t hipMalloc(void **p, size_t bytes)
{
    if (fail_malloc_countdown == 0) { fail_malloc_countdown = -1; *p = (void *)0x1; return hipErrorOutOfMemory; }
    if (0 < fail_malloc_countdown) fail_malloc_countdown--;
    *p = malloc(bytes ? bytes : 1);
    CHECK(*p != NULL);
    for (int i = 0; i < MAX_ALLOCS; i++) {
        if (device_heap[i].base == NULL) {
            device_heap[i].base = *p; device_heap[i].bytes = bytes; device_live++;
            /* fresh device memory is garbage, never zero */
            for (size_t k = 0; k + sizeof(double) <= bytes; k += sizeof(double)) {
                double nan_value = NAN; memcpy((char *)*p + k, &nan_value, sizeof(double));
            }
            return hipSuccess;
        }
    }
    CHECK(0 && "device heap registry full");
    return hipErrorOutOfMemory;
}
static hipError_t hipFree(void *p)
{
    if (p == NULL) return hipSuccess;
    for (int i = 0; i < MAX_ALLOCS; i++) {
        if (device_heap[i].base == p) { device_heap[i].base = NULL; device_live--; free(p); return hipSuccess; }
    }
    CHECK(0 && "hipFree of an unknown pointer");
    return hipSuccess;
}
static hipError_t hipHostMalloc(void **p, size_t bytes, unsigned flags)
{
    (void)flags;
    *p = malloc(bytes ? bytes : 1); CHECK(*p != NULL); pinned_live++; return hipSuccess;
}
static hipError_t hipHostFree(void *p) { if (p) { free(p); pinned_live--; } return hipSuccess; }
static hipError_t hipMemcpyAsync(void *dst, const void *src, size_t bytes, int kind, hipStream_t s)
{
    (void)s;
    if (kind == hipMemcpyHostToDevice || kind == hipMemcpyDeviceToDevice) device_check(dst, bytes, __LINE__);
    if (kind == hipMemcpyDeviceToHost || kind == hipMemcpyDeviceToDevice) device_check(src, bytes, __LINE__);
    memcpy(dst, src, bytes);
    return hipSuccess;
}
static hipError_t hipMemcpy(void *dst, const void *src, size_t bytes, int kind)
{
    return hipMemcpyAsync(dst, src, bytes, kind, NULL);
}
static hipError_t hipMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                                   size_t height, int kind, hipStream_t s)
{
    (void)s; (void)kind;
    CHECK(height == 0 || (width <= dpitch && width <= spitch));
    if (height) {
        device_check(dst, dpitch * (height - 1) + width, __LINE__);
        device_check(src, spitch * (height - 1) + width, __LINE__);
    }
    for (size_t r = 0; r < height; r++) memcpy((char *)dst + r * dpitch, (const char *)src + r * spitch, width);
    return hipSuccess;
}
static hipError_t hipStreamSynchronize(hipStream_t s) { (void)s; return hipSuccess; }
static hipError_t hipStreamCreateWithFlags(hipStream_t *s, unsigned flags)
{
    (void)flags; *s = (hipStream_t)(uintptr_t)0x10; return hipSuccess;
}
static hipError_t hipStreamDestroy(hipStream_t s) { (void)s; return hipSuccess; }
static hipError_t hipGetDevice(int *d) { *d = 0; return hipSuccess; }
static hipError_t hipDeviceSynchronize(void) { return hipSuccess; }
static hipError_t hipGetLastError(void) { return hipSuccess; }
static const char *hipGetErrorString(hipError_t e) { (void)e; return "injected failure"; }
static hipblasStatus_t hipblasCreate(hipblasHandle_t *h) { *h = (hipblasHandle_t)(uintptr_t)0x20; return 0; }
static hipblasStatus_t hipblasDestroy(hipblasHandle_t h) { (void)h; return 0; }
static hipblasStatus_t hipblasSetStream(hipblasHandle_t h, hipStream_t s) { (void)h; (void)s; return 0; }
static void matrix_check(const void *p, int rows, int cols, int ld, size_t elem, int line)
{
    if (rows > 0 && cols > 0) device_check(p, ((size_t)(cols - 1) * (size_t)ld + (size_t)rows) * elem, line);
}
static hipblasStatus_t hipblasDgemm(hipblasHandle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m, int n,
                                    int k, const double *alpha, const double *A, int lda, const double *B, int ldb,
                                    const double *beta, double *C, int ldc)
{
    (void)h;
    matrix_check(A, ta == HIPBLAS_OP_N ? m : k, ta == HIPBLAS_OP_N ? k : m, lda, sizeof(double), __LINE__);
    matrix_check(B, tb == HIPBLAS_OP_N ? k : n, tb == HIPBLAS_OP_N ? n : k, ldb, sizeof(double), __LINE__);
    matrix_check(C, m, n, ldc, sizeof(double), __LINE__);
    dgemm_(ta == HIPBLAS_OP_N ? "N" : "T", tb == HIPBLAS_OP_N ? "N" : "T", &m, &n, &k, alpha, A, &lda, B, &ldb,
           beta, C, &ldc);
    return 0;
}
/* rocBLAS semantics: beta == 0 never reads B (checked on gfx1200) */
static hipblasStatus_t hipblasDgeam(hipblasHandle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m, int n,
                                    const double *alpha, const double *A, int lda, const double *beta,
                                    const double *B, int ldb, double *C, int ldc)
{
    (void)h;
    CHECK(*beta == 0.0 && B == C && tb == HIPBLAS_OP_N && ldb == ldc && A != C);
    matrix_check(A, ta == HIPBLAS_OP_N ? m : n, ta == HIPBLAS_OP_N ? n : m, lda, sizeof(double), __LINE__);
    matrix_check(C, m, n, ldc, sizeof(double), __LINE__);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            C[(size_t)j * ldc + i] = *alpha * (ta == HIPBLAS_OP_N ? A[(size_t)j * lda + i] : A[(size_t)i * lda + j]);
    return 0;
}
static hipblasStatus_t hipblasZgeam(hipblasHandle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m, int n,
                                    const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda,
                                    const hipDoubleComplex *beta, const hipDoubleComplex *B, int ldb,
                                    hipDoubleComplex *C, int ldc)
{
    (void)h;
    CHECK(beta->x == 0.0 && beta->y == 0.0 && B == C && tb == HIPBLAS_OP_N && ldb == ldc && A != C);
    CHECK(alpha->x == 1.0 && alpha->y == 0.0 && ta == HIPBLAS_OP_T);
    matrix_check(A, n, m, lda, sizeof(hipDoubleComplex), __LINE__);
    matrix_check(C, m, n, ldc, sizeof(hipDoubleComplex), __LINE__);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) C[(size_t)j * ldc + i] = A[(size_t)i * lda + j];
    return 0;
}
static hipError_t device_to_host(void *dst, const void *src, size_t bytes)
{
    return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
}
static hipError_t host_to_device(void *dst, const void *src, size_t bytes)
{
    return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
}
#define DEVICE_KIND "host stubs"
#define HAVE_DEVICE_HEAP 1
'''

GPU_DEVICE = r'''
#include "hip_runtime_compat.h"
#include "hipblas_compat.h"
typedef struct { double r, i; } dcomplex;
%(prototypes)s
static hipError_t device_to_host(void *dst, const void *src, size_t bytes)
{
    return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
}
static hipError_t host_to_device(void *dst, const void *src, size_t bytes)
{
    return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
}
static int device_live, pinned_live, fail_malloc_countdown = -1;
#define DEVICE_KIND "HIP/hipBLAS on the GPU"
#define HAVE_DEVICE_HEAP 0
'''

STUBS = r'''
#define DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT 1
static int gemmul8_calls;
static hipblasStatus_t openmx_gemmul8Dgemm(hipblasHandle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m,
                                           int n, int k, const double *alpha, const double *A, int lda,
                                           const double *B, int ldb, const double *beta, double *C, int ldc)
{
    gemmul8_calls++;
    return hipblasDgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
static void openmx_gemmul8ReleaseWorkspaces(void) {}
static int OpenMX_GpuMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
    *free_bytes = (size_t)1 << 34; *total_bytes = (size_t)1 << 34; return 0;
}

/* the dense eigensolvers: LAPACK on a host copy of the device matrix */
static int fault_kind, fault_call, zeig_calls, deig_calls, use_hipsolver, hipsolver_calls, magma_calls;
static int openmx_gpu_eigensolver_use_hipsolver(void) { return use_hipsolver; }
static int host_zheevd(int n, double *a, double *w)
{
    int lwork = -1, lrwork = -1, liwork = -1, info = 0, iq;
    double wq[2], rq;
    zheevd_("V", "L", &n, a, &n, w, wq, &lwork, &rq, &lrwork, &iq, &liwork, &info);
    CHECK(info == 0);
    lwork = (int)wq[0]; lrwork = (int)rq; liwork = iq;
    double *work = malloc(sizeof(double) * 2 * (size_t)lwork), *rwork = malloc(sizeof(double) * (size_t)lrwork);
    int *iwork = malloc(sizeof(int) * (size_t)liwork);
    CHECK(work && rwork && iwork);
    zheevd_("V", "L", &n, a, &n, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
    free(work); free(rwork); free(iwork);
    return info;
}
static int device_zheevd(int n, int maxn, void *d_A, double *w, int *mout)
{
    size_t bytes = sizeof(double) * 2 * (size_t)n * (size_t)n;
    double *a = malloc(bytes), *all = malloc(sizeof(double) * (size_t)n);
    int info;
    CHECK(a && all);
    zeig_calls++;
    *mout = 0;
    CHECK(device_to_host(a, d_A, bytes) == 0);
    info = host_zheevd(n, a, all);
    CHECK(info == 0);
    *mout = maxn;
    if (fault_kind && zeig_calls == fault_call) {
        switch (fault_kind) {
        case 1: info = 37; break;                       /* no convergence */
        case 2: info = -113; break;                     /* MAGMA_ERR_DEVICE_ALLOC */
        case 3: *mout = maxn - 1; break;                /* short eigenpair count */
        case 4: a[2 * (size_t)n * (size_t)(n / 2) + 1] = NAN; break;   /* NaN vector, info 0 */
        case 5: all[n - 1] = NAN; break;               /* NaN eigenvalue, info 0 */
        }
    }
    memcpy(w, all, sizeof(double) * (size_t)maxn);
    CHECK(host_to_device(d_A, a, bytes) == 0);
    free(a); free(all);
    return info;
}
static int openmx_magma_zheevdx_gpu(int n, int maxn, void *d_A, double *w, int *mout)
{
    magma_calls++;
    return device_zheevd(n, maxn, d_A, w, mout);
}
static int openmx_hipsolver_zheevd_gpu(int n, int maxn, void *d_A, double *w, int *mout)
{
    hipsolver_calls++;
    return device_zheevd(n, maxn, d_A, w, mout);
}
static int host_dsyevd(int n, double *a, double *w)
{
    int lwork = -1, liwork = -1, info = 0, iq;
    double wq;
    dsyevd_("V", "L", &n, a, &n, w, &wq, &lwork, &iq, &liwork, &info);
    CHECK(info == 0);
    lwork = (int)wq; liwork = iq;
    double *work = malloc(sizeof(double) * (size_t)lwork);
    int *iwork = malloc(sizeof(int) * (size_t)liwork);
    CHECK(work && iwork);
    dsyevd_("V", "L", &n, a, &n, w, work, &lwork, iwork, &liwork, &info);
    free(work); free(iwork);
    return info;
}
static int openmx_magma_dsyevdx_gpu(int n, int maxn, double *d_A, double *w, int *mout)
{
    size_t bytes = sizeof(double) * (size_t)n * (size_t)n;
    double *a = malloc(bytes), *all = malloc(sizeof(double) * (size_t)n);
    int info;
    CHECK(a && all && maxn == n);
    deig_calls++;
    CHECK(device_to_host(a, d_A, bytes) == 0);
    info = host_dsyevd(n, a, all);
    memcpy(w, all, sizeof(double) * (size_t)maxn);
    *mout = maxn;
    CHECK(host_to_device(d_A, a, bytes) == 0);
    free(a); free(all);
    return info;
}
'''

HARNESS = r'''
static unsigned rng_state = 12345u;
static double rnd(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return (double)(rng_state >> 8) / (double)(1u << 24) - 0.5;
}
static double worst_error;
static void note_error(double e, double limit, const char *what)
{
    if (!(isfinite(e) && e <= limit)) { fprintf(stderr, "FAIL: %s error %.3e > %.1e\n", what, e, limit); exit(2); }
    if (worst_error < e) worst_error = e;
}

/* ------------------------------------------------------------------ */
/* non-collinear                                                       */
/* ------------------------------------------------------------------ */

static double **rows_d(int n2) { double **a = malloc(sizeof(double *) * n2); double *s = calloc((size_t)n2 * n2, sizeof(double)); for (int i = 0; i < n2; i++) a[i] = s + (size_t)i * n2; return a; }
static dcomplex **rows_z(int n2) { dcomplex **a = malloc(sizeof(dcomplex *) * n2); dcomplex *s = calloc((size_t)n2 * n2, sizeof(dcomplex)); for (int i = 0; i < n2; i++) a[i] = s + (size_t)i * n2; return a; }
static void free_rows_d(double **a) { free(a[0]); free(a); }
static void free_rows_z(dcomplex **a) { free(a[0]); free(a); }

/* EigenBand_lapack(C, ko, n, n, 1) of the CPU path: the matrix is C[i][j],
   the vectors come back as C[i][j] = component i of vector j */
static void ref_eigenband(dcomplex **c, double *ko, int n)
{
    double *a = malloc(sizeof(double) * 2 * (size_t)n * n), *w = malloc(sizeof(double) * n);
    for (int i = 1; i <= n; i++) for (int j = 1; j <= n; j++) {
        a[2 * ((size_t)(j - 1) * n + i - 1)] = c[i][j].r; a[2 * ((size_t)(j - 1) * n + i - 1) + 1] = c[i][j].i;
    }
    CHECK(host_zheevd(n, a, w) == 0);
    for (int i = 1; i <= n; i++) for (int j = 1; j <= n; j++) {
        c[i][j].r = a[2 * ((size_t)(j - 1) * n + i - 1)]; c[i][j].i = a[2 * ((size_t)(j - 1) * n + i - 1) + 1];
    }
    for (int i = n; i >= 1; i--) ko[i] = w[i - 1];
    free(a); free(w);
}
/* the superseded GPU hand-off (Eigen_gpusolver_x_complex_openmp): rows of C
   copied as they are into the column-major solver matrix, the solver's
   columns written back as columns of C */
static void legacy_eigen_rowmajor(dcomplex **c, double *ko, int n)
{
    double *a = malloc(sizeof(double) * 2 * (size_t)n * n), *w = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        a[2 * ((size_t)i * n + j)] = c[i + 1][j + 1].r; a[2 * ((size_t)i * n + j) + 1] = c[i + 1][j + 1].i;
    }
    CHECK(host_zheevd(n, a, w) == 0);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        c[j + 1][i + 1].r = a[2 * ((size_t)i * n + j)]; c[j + 1][i + 1].i = a[2 * ((size_t)i * n + j) + 1];
    }
    for (int i = n; i >= 1; i--) ko[i] = w[i - 1];
    free(a); free(w);
}

/* the CPU path of DC_NonCol (P_min == 1 here), loop for loop */
static void cpu_noncol(int NUM, int P_min, double **S_DC, dcomplex **H_DC, dcomplex **C, double *ko, int legacy)
{
    int i1, j1, k, l, jj1, k1, l1, ii1, NUM1;
    double tmp1, tmp2, sum_r, sum_i, sum1_r, sum1_i, sum2_r, sum2_i;

    for (i1 = 1; i1 <= NUM; i1++) for (j1 = i1 + 1; j1 <= NUM; j1++) {
        tmp1 = S_DC[i1][j1]; tmp2 = S_DC[j1][i1]; S_DC[i1][j1] = tmp2; S_DC[j1][i1] = tmp1;
    }
    for (j1 = P_min; j1 <= NUM; j1++) {
        for (k = 0; k <= 1; k++) {
            jj1 = 2 * j1 - P_min + k;
            k1  = k * NUM;
            for (i1 = 1; i1 <= 2 * NUM; i1++) {
                sum_r = 0.0; sum_i = 0.0;
                for (l = 1; l <= NUM; l++) {
                    l1 = k1 + l;
                    sum_r += H_DC[i1][l1].r * S_DC[j1][l];
                    sum_i += H_DC[i1][l1].i * S_DC[j1][l];
                }
                C[jj1][i1].r = sum_r; C[jj1][i1].i = sum_i;
            }
        }
    }
    for (j1 = 1; j1 <= 2 * NUM; j1++) {
        for (i1 = P_min; i1 <= NUM; i1++) {
            sum1_r = 0.0; sum1_i = 0.0; sum2_r = 0.0; sum2_i = 0.0;
            for (l = 1; l <= NUM; l++) { sum1_r += S_DC[i1][l] * C[j1][l].r; sum1_i += S_DC[i1][l] * C[j1][l].i; }
            for (l = NUM + 1; l <= 2 * NUM; l++) {
                l1 = l - NUM;
                sum2_r += S_DC[i1][l1] * C[j1][l].r; sum2_i += S_DC[i1][l1] * C[j1][l].i;
            }
            ii1 = 2 * i1 - P_min;     H_DC[j1][ii1].r = sum1_r; H_DC[j1][ii1].i = sum1_i;
            ii1 = 2 * i1 - P_min + 1; H_DC[j1][ii1].r = sum2_r; H_DC[j1][ii1].i = sum2_i;
        }
    }
    for (i1 = P_min; i1 <= 2 * NUM; i1++) for (j1 = P_min; j1 <= 2 * NUM; j1++) {
        C[j1 - (P_min - 1)][i1 - (P_min - 1)].r = H_DC[i1][j1].r;
        C[j1 - (P_min - 1)][i1 - (P_min - 1)].i = H_DC[i1][j1].i;
    }
    NUM1 = 2 * NUM - (P_min - 1);
    if (legacy) legacy_eigen_rowmajor(C, ko, NUM1);
    else        ref_eigenband(C, ko, NUM1);
    for (i1 = 1; i1 <= NUM1; i1++) for (j1 = 1; j1 <= NUM1; j1++) {
        H_DC[j1][i1].r = C[i1][j1].r; H_DC[j1][i1].i = C[i1][j1].i;
    }
    for (i1 = 1; i1 <= NUM; i1++) for (j1 = i1 + 1; j1 <= NUM; j1++) {
        tmp1 = S_DC[i1][j1]; tmp2 = S_DC[j1][i1]; S_DC[i1][j1] = tmp2; S_DC[j1][i1] = tmp1;
    }
    for (j1 = 1; j1 <= NUM1; j1++) for (k = 0; k <= 1; k++) for (i1 = 1; i1 <= NUM; i1++) {
        sum_r = 0.0; sum_i = 0.0;
        for (l = P_min; l <= NUM; l++) {
            sum_r += S_DC[i1][l] * H_DC[j1][2 * (l - P_min) + 1 + k].r;
            sum_i += S_DC[i1][l] * H_DC[j1][2 * (l - P_min) + 1 + k].i;
        }
        C[i1 + k * NUM][j1].r = sum_r; C[i1 + k * NUM][j1].i = sum_i;
    }
}

/* overlap S (NUM x NUM, SPD) and its transform T = U lambda^-1/2 as the
   SCF_iter <= 2 block of DC_NonCol leaves it in S_DC */
static void make_overlap(int NUM, double *s, double **T)
{
    double *a = malloc(sizeof(double) * (size_t)NUM * NUM), *w = malloc(sizeof(double) * NUM);
    for (int i = 0; i < NUM; i++) for (int j = 0; j <= i; j++) {
        double v = (i == j) ? 1.0 + 0.3 * fabs(rnd()) : 0.12 * rnd() / (1.0 + 0.2 * (i - j));
        s[(size_t)i * NUM + j] = v; s[(size_t)j * NUM + i] = v;
    }
    memcpy(a, s, sizeof(double) * (size_t)NUM * NUM);
    CHECK(host_dsyevd(NUM, a, w) == 0);
    for (int j = 0; j < NUM; j++) CHECK(w[j] > 1e-3);
    for (int i = 1; i <= NUM; i++) for (int j = 1; j <= NUM; j++)
        T[i][j] = a[(size_t)(j - 1) * NUM + (i - 1)] / sqrt(w[j - 1]);
    free(a); free(w);
}
/* spinor Hamiltonian as DC_NonCol builds it: kind 0 real (no SOC), 1
   magnetic (complex up-down block), 2 spin-orbit-like (complex diagonal
   blocks), 3 non-magnetic (block diagonal with equal blocks: degenerate) */
static void make_hamiltonian(int NUM, int kind, dcomplex **H)
{
    for (int i = 1; i <= NUM; i++) for (int j = 1; j <= i; j++) {
        double huu = (i == j ? 2.0 * rnd() : rnd() / (1.0 + 0.1 * (i - j)));
        double hdd = (kind == 3) ? huu : huu + 0.3 * rnd();
        double iuu = (kind == 2 && i != j) ? 0.2 * rnd() : 0.0;
        double idd = (kind == 2 && i != j) ? 0.2 * rnd() : 0.0;
        H[i][j].r = huu; H[i][j].i = iuu; H[j][i].r = huu; H[j][i].i = -iuu;
        H[i + NUM][j + NUM].r = hdd; H[i + NUM][j + NUM].i = idd;
        H[j + NUM][i + NUM].r = hdd; H[j + NUM][i + NUM].i = -idd;
    }
    for (int i = 1; i <= NUM; i++) for (int j = 1; j <= NUM; j++) {
        double r = (kind == 3) ? 0.0 : 0.15 * rnd(), im = (kind == 1 || kind == 2) ? 0.15 * rnd() : 0.0;
        H[i][j + NUM].r = r; H[i][j + NUM].i = im;
        H[j + NUM][i].r = r; H[j + NUM][i].i = -im;
    }
}
/* D(f) = sum_s f(e_s) c_s c_s^H with a smooth occupation: invariant under
   phases and rotations inside degenerate levels */
static void density(int dim, dcomplex **C, const double *ko, double mu, double kt, int energy, dcomplex *D)
{
    memset(D, 0, sizeof(dcomplex) * (size_t)dim * dim);
    for (int s = 1; s <= dim; s++) {
        double f = 1.0 / (1.0 + exp((ko[s] - mu) / kt));
        if (energy) f *= ko[s];
        for (int a = 1; a <= dim; a++) for (int b = 1; b <= dim; b++) {
            D[(size_t)(a - 1) * dim + b - 1].r += f * (C[a][s].r * C[b][s].r + C[a][s].i * C[b][s].i);
            D[(size_t)(a - 1) * dim + b - 1].i += f * (C[a][s].i * C[b][s].r - C[a][s].r * C[b][s].i);
        }
    }
}
static double max_diff(int n, const dcomplex *x, const dcomplex *y, int imag_only)
{
    double m = 0.0;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        double d = imag_only ? fabs(x[i].i - y[i].i) : hypot(x[i].r - y[i].r, x[i].i - y[i].i);
        if (m < d) m = d;
    }
    return m;
}

static int noncol_checks, noncol_failures;

static void noncol_case(int NUM, int kind, int hipsolver)
{
    const int n2 = 2 * NUM + 3, dim = 2 * NUM;
    double *s = malloc(sizeof(double) * (size_t)NUM * NUM);
    double **T = rows_d(NUM + 2), **T_cpu = rows_d(NUM + 2), **T0 = rows_d(NUM + 2);
    dcomplex **H = rows_z(n2), **H_cpu = rows_z(n2), **H0 = rows_z(n2), **C = rows_z(n2), **C_cpu = rows_z(n2);
    double *ko = calloc(n2, sizeof(double)), *ko_cpu = calloc(n2, sizeof(double));
    dcomplex *D = malloc(sizeof(dcomplex) * (size_t)dim * dim), *D_cpu = malloc(sizeof(dcomplex) * (size_t)dim * dim);
    double mu, kt = 0.15, e;

    make_overlap(NUM, s, T);
    make_hamiltonian(NUM, kind, H);
    memcpy(T0[0], T[0], sizeof(double) * (size_t)(NUM + 2) * (NUM + 2));
    memcpy(H0[0], H[0], sizeof(dcomplex) * (size_t)n2 * n2);

    use_hipsolver = hipsolver;
    CHECK(DCNonCol_GpuSolve(NUM, NUM + 3, T, H, C, ko) == 1);
    CHECK(memcmp(T[0], T0[0], sizeof(double) * (size_t)(NUM + 2) * (NUM + 2)) == 0);
    CHECK(memcmp(H[0], H0[0], sizeof(dcomplex) * (size_t)n2 * n2) == 0);

    memcpy(T_cpu[0], T0[0], sizeof(double) * (size_t)(NUM + 2) * (NUM + 2));
    memcpy(H_cpu[0], H0[0], sizeof(dcomplex) * (size_t)n2 * n2);
    cpu_noncol(NUM, 1, T_cpu, H_cpu, C_cpu, ko_cpu, 0);

    for (int i = 1; i <= dim; i++) note_error(fabs(ko[i] - ko_cpu[i]), 1e-11, "eigenvalue");
    mu = 0.5 * (ko_cpu[NUM] + ko_cpu[NUM + 1]);
    for (int energy = 0; energy <= 1; energy++) {
        density(dim, C, ko, mu, kt, energy, D);
        density(dim, C_cpu, ko_cpu, mu, kt, energy, D_cpu);
        note_error(max_diff(dim, D, D_cpu, 0), 1e-10, "density matrix");
    }
    /* H C = diag(S, S) C E for the GPU vectors */
    e = 0.0;
    for (int s1 = 1; s1 <= dim; s1++) for (int a = 1; a <= dim; a++) {
        double hr = 0, hi = 0, sr = 0, si = 0;
        for (int b = 1; b <= dim; b++) {
            hr += H0[a][b].r * C[b][s1].r - H0[a][b].i * C[b][s1].i;
            hi += H0[a][b].r * C[b][s1].i + H0[a][b].i * C[b][s1].r;
        }
        int blk = (a > NUM), ai = a - blk * NUM;
        for (int b = 1; b <= NUM; b++) {
            sr += s[(size_t)(ai - 1) * NUM + b - 1] * C[b + blk * NUM][s1].r;
            si += s[(size_t)(ai - 1) * NUM + b - 1] * C[b + blk * NUM][s1].i;
        }
        double d = hypot(hr - ko[s1] * sr, hi - ko[s1] * si);
        if (e < d) e = d;
    }
    note_error(e, 1e-10, "generalized residual");

    /* the old row-major hand-off: conjugate vectors */
    if (kind == 1 || kind == 2) {
        double legacy;
        memcpy(T_cpu[0], T0[0], sizeof(double) * (size_t)(NUM + 2) * (NUM + 2));
        memcpy(H_cpu[0], H0[0], sizeof(dcomplex) * (size_t)n2 * n2);
        cpu_noncol(NUM, 1, T_cpu, H_cpu, C_cpu, ko_cpu, 1);
        density(dim, C_cpu, ko_cpu, mu, kt, 0, D_cpu);
        density(dim, C, ko, mu, kt, 0, D);
        legacy = max_diff(dim, D, D_cpu, 1);
        CHECK(legacy > 1e-3);
        printf("  NUM=%d kind=%d: the superseded row-major GPU hand-off misses Im D by %.3g "
               "(it diagonalized conj(H'))\n", NUM, kind, legacy);
    }
    noncol_checks++;

    free(s); free_rows_d(T); free_rows_d(T_cpu); free_rows_d(T0);
    free_rows_z(H); free_rows_z(H_cpu); free_rows_z(H0); free_rows_z(C); free_rows_z(C_cpu);
    free(ko); free(ko_cpu); free(D); free(D_cpu);
}

static void noncol_fault(int NUM, int kind)
{
    const int n2 = 2 * NUM + 3;
    double *s = malloc(sizeof(double) * (size_t)NUM * NUM);
    double **T = rows_d(NUM + 2), **T0 = rows_d(NUM + 2);
    dcomplex **H = rows_z(n2), **H0 = rows_z(n2), **C = rows_z(n2);
    double *ko = calloc(n2, sizeof(double));

    make_overlap(NUM, s, T);
    make_hamiltonian(NUM, 1, H);
    memcpy(T0[0], T[0], sizeof(double) * (size_t)(NUM + 2) * (NUM + 2));
    memcpy(H0[0], H[0], sizeof(dcomplex) * (size_t)n2 * n2);

    DC_noncol_gpu.off = 0;
    if (kind == 6) {
        /* the device allocation of a larger capacity fails */
        DCNonCol_GpuRelease(1);
        fail_malloc_countdown = HAVE_DEVICE_HEAP ? 1 : -1;
        if (!HAVE_DEVICE_HEAP) { free(s); free_rows_d(T); free_rows_d(T0); free_rows_z(H); free_rows_z(H0); free_rows_z(C); free(ko); return; }
    } else {
        fault_kind = kind; fault_call = zeig_calls + 1;
    }
    CHECK(DCNonCol_GpuSolve(NUM, NUM, T, H, C, ko) == 0);
    fault_kind = 0;
    CHECK(memcmp(T[0], T0[0], sizeof(double) * (size_t)(NUM + 2) * (NUM + 2)) == 0);
    CHECK(memcmp(H[0], H0[0], sizeof(dcomplex) * (size_t)n2 * n2) == 0);
    /* allocation failures keep the rest of the call on the CPU */
    CHECK(DC_noncol_gpu.off == (kind == 2 || kind == 6));
    if (DC_noncol_gpu.off) {
        int calls = zeig_calls;
        CHECK(DCNonCol_GpuSolve(NUM, NUM, T, H, C, ko) == 0 && zeig_calls == calls);
        DC_noncol_gpu.off = 0;   /* the next DC_NonCol call is admitted afresh */
    }
    CHECK(DCNonCol_GpuSolve(NUM, NUM, T, H, C, ko) == 1);
    noncol_failures++;

    free(s); free_rows_d(T); free_rows_d(T0); free_rows_z(H); free_rows_z(H0); free_rows_z(C); free(ko);
}

/* ------------------------------------------------------------------ */
/* collinear                                                           */
/* ------------------------------------------------------------------ */

static int col_checks;

/* the previous host-packed staging (PackMatrix/UnpackMatrix), on the same
   device and with the same GEMMs and eigensolver */
static void old_pack(int n, double **src, double *dst)
{
    for (int j = 1; j <= n; j++) for (int i = 1; i <= n; i++) dst[(size_t)(j - 1) * n + i - 1] = src[i][j];
}
static void old_solve(int n, int p_min, double **S, double **Hs, double *ko, double **C)
{
    hipblasHandle_t h = DC_gpusolver_ctx.hipblas;
    int num1 = n - (p_min - 1), mout = 0;
    double one = 1.0, zero = 0.0, *host = malloc(sizeof(double) * (size_t)n * n), *dS, *dH, *dT, *dA, *dC;
    size_t bytes = sizeof(double) * (size_t)n * n;
    CHECK(hipMalloc((void **)&dS, bytes) == 0 && hipMalloc((void **)&dH, bytes) == 0 &&
          hipMalloc((void **)&dT, bytes) == 0 && hipMalloc((void **)&dA, bytes) == 0 &&
          hipMalloc((void **)&dC, bytes) == 0);
    old_pack(n, S, host); CHECK(host_to_device(dS, host, bytes) == 0);
    old_pack(n, Hs, host); CHECK(host_to_device(dH, host, bytes) == 0);
    CHECK(hipblasDgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, &one, dH, n, dS, n, &zero, dT, n) == 0);
    CHECK(hipblasDgemm(h, HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n, &one, dS, n, dT, n, &zero, dH, n) == 0);
    CHECK(hipMemcpy2DAsync(dA, sizeof(double) * num1, dH + (size_t)(p_min - 1) * n + (p_min - 1),
                           sizeof(double) * n, sizeof(double) * num1, num1, hipMemcpyDeviceToDevice,
                           DC_gpusolver_ctx.stream) == 0);
    CHECK(hipStreamSynchronize(DC_gpusolver_ctx.stream) == 0);
    CHECK(openmx_magma_dsyevdx_gpu(num1, num1, dA, ko + 1, &mout) == 0 && mout == num1);
    CHECK(hipblasDgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, n, num1, num1, &one, dS + (size_t)(p_min - 1) * n, n, dA,
                       num1, &zero, dC, n) == 0);
    CHECK(hipStreamSynchronize(DC_gpusolver_ctx.stream) == 0);
    CHECK(device_to_host(host, dC, sizeof(double) * (size_t)n * num1) == 0);
    for (int j = 1; j <= num1; j++) for (int i = 1; i <= n; i++) C[i][j] = host[(size_t)(j - 1) * n + i - 1];
    hipFree(dS); hipFree(dH); hipFree(dT); hipFree(dA); hipFree(dC); free(host);
}
static void random_symmetric(int n, double **a, double diag)
{
    for (int i = 1; i <= n; i++) for (int j = 1; j <= i; j++) {
        double v = (i == j) ? diag + rnd() : 0.2 * rnd();
        a[i][j] = v; a[j][i] = v * (1.0 + (i != j ? 1e-15 * (i - j) : 0.0));  /* not bitwise symmetric */
    }
}
static int same_rows(int nr, int nc, double **x, double **y)
{
    for (int i = 1; i <= nr; i++) if (memcmp(&x[i][1], &y[i][1], sizeof(double) * nc) != 0) return 0;
    return 1;
}

static void col_case(int n, int p_min, int slot_mode)
{
    const int n2 = n + 3, num1 = n - (p_min - 1);
    double **S = rows_d(n2), **S0 = rows_d(n2), **H = rows_d(n2), **H0 = rows_d(n2);
    double **C = rows_d(n2), **C_old = rows_d(n2), *ko = calloc(n2, sizeof(double)), *ko_old = calloc(n2, sizeof(double));
    const double *d_S_use = NULL;

    DC_GpuSolver_Init();
    random_symmetric(n, S, 1.0);
    random_symmetric(n, H, 0.0);
    memcpy(S0[0], S[0], sizeof(double) * (size_t)n2 * n2);
    memcpy(H0[0], H[0], sizeof(double) * (size_t)n2 * n2);

    /* the previous staging */
    old_solve(n, p_min, S, H, ko_old, C_old);

    /* DC_Col: stage the transformed overlap, keep a resident copy when the
       atom has a slot, solve */
    if (slot_mode) {
        Divide_Conquer_Release_GPU_SCache();
        DC_scache.matomnum = 1;
        DC_scache.dim = calloc(2, sizeof(int)); DC_scache.valid = calloc(2, sizeof(int));
        DC_scache.off = calloc(2, sizeof(size_t));
        DC_scache.dim[1] = n;
        DC_scache.off[1] = (slot_mode == 2) ? 0 : DC_SCACHE_SLOT_ALIGN;   /* behind a first slot */
        DC_scache.arena_elems = DC_scache.off[1] + (size_t)n * n;
        CHECK(hipMalloc((void **)&DC_scache.d_arena, sizeof(double) * DC_scache.arena_elems) == 0);
        DC_scache.state = 1;
    }
    CHECK(DCCol_GpuSolver_LoadTransformedOverlap(n, S));
    CHECK(DCCol_SCache_Store(1, n, &d_S_use));
    CHECK((d_S_use != NULL) == (slot_mode != 0));
    CHECK(DCCol_GpuSolver_SolveHamiltonian(n, p_min, d_S_use, S, H, ko, C));
    CHECK(memcmp(ko + 1, ko_old + 1, sizeof(double) * num1) == 0);
    CHECK(same_rows(n, num1, C, C_old));
    /* the second spin of the same atom: overlap still resident or reloaded */
    memset(C[0], 0, sizeof(double) * (size_t)n2 * n2);
    CHECK(DCCol_GpuSolver_SolveHamiltonian(n, p_min, d_S_use, S, H, ko, C));
    CHECK(same_rows(n, num1, C, C_old));
    if (slot_mode) {
        /* a later SCF step: S-cache hit, the host S_DC is not even filled */
        double **S_empty = rows_d(n2);
        memset(C[0], 0, sizeof(double) * (size_t)n2 * n2);
        DC_GpuSolver_ReleaseOverlapBuffer();
        CHECK(DCCol_SCache_ValidPtr(1, n) == d_S_use);
        CHECK(DCCol_GpuSolver_SolveHamiltonian(n, p_min, DCCol_SCache_ValidPtr(1, n), S_empty, H, ko, C));
        CHECK(same_rows(n, num1, C, C_old));
        free_rows_d(S_empty);
        Divide_Conquer_Release_GPU_SCache();
    }
    CHECK(memcmp(S[0], S0[0], sizeof(double) * (size_t)n2 * n2) == 0);
    CHECK(memcmp(H[0], H0[0], sizeof(double) * (size_t)n2 * n2) == 0);

    /* the overlap diagonalization of the first SCF steps */
    {
        double *host = malloc(sizeof(double) * (size_t)n * n), *w = malloc(sizeof(double) * n), *ko2 = calloc(n2, sizeof(double));
        old_pack(n, S0, host);
        CHECK(host_dsyevd(n, host, w) == 0);
        CHECK(DCCol_GpuSolver_DiagonalizeOverlap(n, S, ko2));
        CHECK(memcmp(ko2 + 1, w, sizeof(double) * n) == 0);
        for (int i = 1; i <= n; i++) for (int j = 1; j <= n; j++)
            CHECK(memcmp(&S[i][j], &host[(size_t)(j - 1) * n + i - 1], sizeof(double)) == 0);
        free(host); free(w); free(ko2);
    }
    col_checks++;
    free_rows_d(S); free_rows_d(S0); free_rows_d(H); free_rows_d(H0); free_rows_d(C); free_rows_d(C_old);
    free(ko); free(ko_old);
}

int main(void)
{
    const int noncol_sizes[] = {3, 8, 23};
    const int col_sizes[] = {40, 31, 37, 52, 45, 52, 9};

    printf("device: %s\n", DEVICE_KIND);
    for (int a = 0; a < 3; a++)
        for (int kind = 0; kind < 4; kind++)
            for (int hs = 0; hs <= 1; hs++) noncol_case(noncol_sizes[a], kind, hs);
    for (int kind = 1; kind <= 6; kind++) noncol_fault(12, kind);
    CHECK(0 < magma_calls && 0 < hipsolver_calls);
    DCNonCol_GpuRelease(0);

    for (int a = 0; a < 7; a++) {
        const int n = col_sizes[a];
        col_case(n, 1, a % 3);
        col_case(n, 3, (a + 1) % 3);
    }
    /* run boundary: the context, the staging buffers and the S-cache go,
       the failure flags return to their build defaults, and the next input
       re-creates everything on demand */
    DC_gpusolver_gemm_disabled = 1; DC_gpusolver_eigen_disabled = 1;
    DC_gpusolver_gemmul8_disabled = 0; DC_gpusolver_approved_gemm_n = 77; DC_gpusolver_native_approved = 1;
    Divide_Conquer_Release_GPU_Caches();
    CHECK(!DC_gpusolver_ctx.initialized && DC_gpusolver_ctx.hipblas == NULL && DC_gpusolver_ctx.stream == NULL);
    CHECK(DC_gpusolver_ctx.h_matrix == NULL && DC_gpusolver_ctx.d_S == NULL && DC_gpusolver_ctx.matrix_dim == 0);
    CHECK(DC_noncol_gpu.h_stage == NULL && DC_noncol_gpu.d_A == NULL && DC_scache.d_arena == NULL);
    CHECK(!DC_gpusolver_gemm_disabled && !DC_gpusolver_eigen_disabled && DC_gpusolver_gemmul8_disabled == 1);
    CHECK(DC_gpusolver_approved_gemm_n == 0 && !DC_gpusolver_native_approved);
    CHECK(device_live == 0 && pinned_live == 0);
    Divide_Conquer_Release_GPU_Caches();   /* idempotent */
    col_case(33, 1, 1);
    noncol_case(5, 1, 0);
    Divide_Conquer_Release_GPU_Caches();
    CHECK(device_live == 0 && pinned_live == 0);
    printf("PASS: %d non-collinear solves (real, magnetic, spin-orbit-like, degenerate; MAGMA and hipSOLVER "
           "dispatch) match the CPU path, %d injected failures leave S and H intact; %d collinear "
           "staging/S-cache cases are bit-identical to the previous staging; the run-boundary release "
           "frees everything and resets the flags, the next input re-creates the context; max error %.3g\n",
           noncol_checks, noncol_failures, col_checks, worst_error);
    return 0;
}
'''


def make_harness(source, gpu):
    prototypes = "\n".join(prototype(source, name) for name in ["hipblasDgeam", "hipblasZgeam"])
    parts = [COMMON, (GPU_DEVICE % {"prototypes": prototypes}) if gpu else CPU_DEVICE, STUBS]
    parts += [define(source, name) for name in
              ["DC_MAGMA_ERR_HOST_ALLOC", "DC_MAGMA_ERR_DEVICE_ALLOC", "DC_SCACHE_SLOT_ALIGN"]]
    parts += [function(source, name) for name in
              ["DC_CheckedMulCount", "DC_CheckedArrayBytes", "DC_EnvU32"]]
    parts += [typedef(source, "DCGpuSolverCtx"),
              "static DCGpuSolverCtx DC_gpusolver_ctx = {0};",
              "static int DC_gpusolver_gemm_disabled = 0;",
              "static int DC_gpusolver_gemmul8_disabled = DC_GPUSOLVER_GEMMUL8_DISABLED_DEFAULT;",
              "static int DC_gpusolver_eigen_disabled = 0;",
              "static int DC_gpusolver_native_approved = 0;",
              "static int DC_gpusolver_approved_gemm_n = 0;",
              "static int DC_gpusolver_approved_gemm_num1 = 0;",
              typedef(source, "DCColSCacheCtx"),
              "static DCColSCacheCtx DC_scache = {0};",
              public(source, "Divide_Conquer_Release_GPU_SCache"),
              "static void DC_GpuSolver_Destroy(void);",
              "static void DCNonCol_GpuRelease(int keep_stage);",
              "static void DC_GpuSolver_DisableGemmPathHip(const char *where, hipError_t status);"]
    parts += [function(source, name) for name in
              ["DCCol_SCache_Ptr", "DCCol_SCache_ValidPtr", "DC_GpuSolver_HipblasStatusName",
               "DC_GpuSolver_GemmDisabled", "DC_GpuSolver_EigenDisabled", "DC_GpuSolver_HipblasReserveBytes",
               "DC_GpuSolver_ReleaseGemmScratch", "DC_GpuSolver_ReleaseTransformScratch",
               "DC_GpuSolver_ReleaseOverlapBuffer", "DC_GpuSolver_DisableEigenPathInfo",
               "DC_GpuSolver_DisableEigenPathEigenpairs", "DC_GpuSolver_HasHipblasMemoryForSolve",
               "DC_GpuSolver_DisableGemmPath", "DC_GpuSolver_DisableGemmPathHip", "DC_GpuSolver_TryGpuDgemm",
               "DC_GpuSolver_Destroy", "DC_GpuSolver_Init", "DC_GpuSolver_EnsureMatrixCapacity",
               "DC_GpuSolver_AllocMatrix", "DC_GpuSolver_EnsureGemmCapacity", "DC_GpuSolver_Eigen",
               "DCCol_GpuSolver_PackRows", "DCCol_GpuSolver_UnpackRows", "DC_GpuSolver_DeviceTranspose",
               "DCCol_GpuSolver_UploadMatrix", "DCCol_GpuSolver_LoadTransformedOverlap", "DCCol_SCache_Store",
               "DCCol_GpuSolver_DiagonalizeOverlap", "DCCol_GpuSolver_SolveHamiltonian"]]
    parts += [typedef(source, "DCNonColGpuCtx"), "static DCNonColGpuCtx DC_noncol_gpu = {0};"]
    parts += [function(source, name) for name in
              ["DCNonCol_GpuRelease", "DCNonCol_StageOffset", "DCNonCol_ComplexBytes", "DCNonCol_GpuEnsure",
               "DCNonCol_GpuReport", "DCNonCol_GpuTranspose", "DCNonCol_GpuTimesT", "DCNonCol_GpuSolve"]]
    parts += [public(source, "Divide_Conquer_Release_GPU_Caches")]
    return "\n\n".join(parts + [HARNESS])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"),
                        help="C compiler of the host-stub build (default: $CC or cc)")
    parser.add_argument("--gpu", action="store_true",
                        help="build against the real HIP runtime and hipBLAS and run on the GPU")
    parser.add_argument("--rocm", default=os.environ.get("ROCM_PATH", "/opt/rocm-7.2.0"))
    parser.add_argument("--sanitize", action="store_true", help="host-stub build with ASan/UBSan")
    parser.add_argument("--libs", default=os.environ.get("LAPACK_LIBS", "-llapack -lblas"),
                        help="LP64 CPU LAPACK/BLAS libraries (default: -llapack -lblas)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / "source/Divide_Conquer.c").read_text()
    flags = shlex.split(os.environ.get("CFLAGS", "-O2"))
    with tempfile.TemporaryDirectory(prefix="openmx-dc-gpu-check-") as directory:
        directory = Path(directory)
        harness = directory / "harness.c"
        executable = directory / "harness"
        harness.write_text(make_harness(source, args.gpu))
        if args.gpu:
            cc = str(Path(args.rocm) / "llvm/bin/clang")
            command = [cc, *flags, "-std=c11", "-D__HIP_PLATFORM_AMD__", f"-I{root / 'source'}",
                       str(harness), "-o", str(executable), f"-L{Path(args.rocm) / 'lib'}",
                       f"-Wl,-rpath,{Path(args.rocm) / 'lib'}", "-lhipblas", "-lamdhip64",
                       *shlex.split(args.libs), "-lm"]
        else:
            command = [args.cc, *flags, "-std=c11", "-D__HIP_PLATFORM_AMD__", str(harness), "-o",
                       str(executable), *shlex.split(args.libs), "-lm"]
            if args.sanitize:
                command[1:1] = ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        subprocess.run(command, check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
