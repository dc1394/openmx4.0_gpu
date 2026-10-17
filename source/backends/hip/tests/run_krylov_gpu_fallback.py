#!/usr/bin/env python3
"""Check the GPU helpers of the Krylov O(N) solver without a GPU or MPI.

The production helpers of source/Krylov.c (the per-thread workspace, the
per-call Krylov_Dgemm/Krylov_Eigen2, the fused Krylov_ProjectedSolve_GPU,
the KU cache and the device-memory preflight) are extracted verbatim and
built against host stubs of HIP, hipBLAS, the GEMMul8 bridge, MAGMA and MPI
(one rank).  Device buffers are host memory and the device GEMMs and
eigensolves are LAPACK/BLAS, so the results can be compared with the CPU
sequence of Krylov_Col (Krylov_Dgemm/Krylov_Eigen2 without a workspace).

Checked:
  * the fused path and the per-call path agree with the CPU sequence
    (eigenvalues, and the eigenvector rows of C up to their sign), with
    and without the KU cache;
  * OPENMX_KRYLOV_GEMMUL8 routes the device products through
    openmx_gemmul8Dgemm (default) or hipblasDgemm (0);
  * the eigensolver dispatch of the fused path: LAPACK below
    Krylov_gpu_eigen_min (MAGMA is not called), MAGMA from it on, LAPACK
    again after a MAGMA failure;
  * a device GEMM that returns an error (each of the three products of the
    fused path, with the eigenvectors on the device or on the host, and of
    the per-call path) is redone on the CPU with the same result, and the
    rest of the call stays on the CPU;
  * the preflight adds the GEMMul8 workspace of the largest product on the
    GPU once per thread of the pool, returns the workspaces held before it
    measures the free memory (after the device barrier), and leaves both
    out when the products are native, when GEMMul8 is off, or when no
    product reaches the GPU.
This checks the control flow and the numbers, not HIP or GEMMul8.
--sanitize adds ASan/UBSan.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def function(source, name, static=True):
    prefix = r"^static [^;{}]*\b" if static else r"^[a-z][^;{}()]*\b"
    match = re.search(prefix + re.escape(name) + r"\([^;]*?\)\s*\{", source, re.MULTILINE)
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


def variable(source, name):
    match = re.search(r"^static [^;(]*\b" + re.escape(name) + r"\b[^;(]*;", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production variable {name}")
    return match.group(0)


def typedef(source, name):
    end = source.index("} " + name + ";") + len("} " + name + ";")
    return source[source.rfind("typedef struct {", 0, end):end]


def between(source, first, last):
    start = source.index(first)
    return source[start:source.index(last, start) + len(last)]


PREFIX = r'''
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); \
} } while (0)
typedef int hipError_t;
typedef int hipblasStatus_t;
typedef void *hipStream_t;
typedef void *hipblasHandle_t;
typedef enum { HIPBLAS_OP_N = 111, HIPBLAS_OP_T = 112, HIPBLAS_OP_C = 113 } hipblasOperation_t;
enum { hipSuccess = 0, hipStreamNonBlocking = 1, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };
enum { HIPBLAS_STATUS_SUCCESS = 0, HIPBLAS_STATUS_ALLOC_FAILED = 2, HIPBLAS_STATUS_EXECUTION_FAILED = 6 };
#define F77_NAME(lower, upper) lower ## _
extern void dgemm_(const char *, const char *, const int *, const int *, const int *,
                   const double *, const double *, const int *, const double *,
                   const int *, const double *, double *, const int *);
extern void dsyevd_(const char *, const char *, const int *, double *, const int *,
                    double *, double *, const int *, int *, const int *, int *);
#define wait_hipfunc(call) CHECK((call) == 0)

/* MPI, one rank */
typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Info;
#define MPI_COMM_NULL (-1)
#define MPI_COMM_WORLD 0
#define MPI_INFO_NULL 0
#define MPI_COMM_TYPE_SHARED 0
#define MPI_UNSIGNED_LONG_LONG 1
#define MPI_SUM 1
#define MPI_MIN 2
static MPI_Comm mpi_comm_level1 = 0;
static int Host_ID = 0, level_stdout = 1;
static char events[256];
static void event(char c) { size_t n = strlen(events); CHECK(n + 1 < sizeof(events)); events[n] = c; events[n+1] = 0; }
static int MPI_Comm_split_type(MPI_Comm c, int t, int k, MPI_Info i, MPI_Comm *out) { *out = 1; return 0; }
static int MPI_Comm_split(MPI_Comm c, int color, int key, MPI_Comm *out) { *out = 2; return 0; }
static int MPI_Comm_free(MPI_Comm *c) { *c = MPI_COMM_NULL; return 0; }
static int MPI_Comm_size(MPI_Comm c, int *size) { *size = 1; return 0; }
static int MPI_Comm_rank(MPI_Comm c, int *rank) { *rank = 0; return 0; }
static int MPI_Allreduce(const void *s, void *r, int count, MPI_Datatype t, MPI_Op op, MPI_Comm c)
{
    CHECK(count == 1 && t == MPI_UNSIGNED_LONG_LONG);
    memcpy(r, s, sizeof(unsigned long long)); return 0;
}
static int MPI_Barrier(MPI_Comm c) { event('B'); return 0; }
static int MPI_Abort(MPI_Comm c, int code) { fprintf(stderr, "MPI_Abort\n"); exit(4); }

/* OpenMX globals */
#define GPUSOLVER 7
static int scf_eigen_lib_flag = GPUSOLVER;
static int openmx_band_gpu_dense_oversubscribed(void) { return 0; }
static void dtime(double *t) { *t = 0.0; }

/* HIP */
static int device_allocations;
static size_t free_memory = (size_t)8 << 30;
static int hipFree(void *p) { if (p) { free(p); device_allocations--; } return 0; }
static int hipMalloc(void **p, size_t bytes)
{
    *p = malloc(bytes ? bytes : 1); CHECK(*p != NULL); device_allocations++; return 0;
}
static int hipMemcpyAsync(void *to, const void *from, size_t bytes, int kind, hipStream_t s)
{
    memcpy(to, from, bytes); return 0;
}
static int hipStreamSynchronize(hipStream_t s) { return 0; }
static int hipStreamCreateWithFlags(hipStream_t *s, int flags) { *s = (void *)1; return 0; }
static int hipStreamDestroy(hipStream_t s) { return 0; }
static int hipGetLastError(void) { return 0; }
static int hipGetDevice(int *d) { *d = 0; return 0; }
static const char *hipGetErrorString(int e) { return "stub error"; }
static int OpenMX_GpuMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
    event('M'); *free_bytes = free_memory; *total_bytes = (size_t)16 << 30; return 0;
}
static int hipblasCreate(hipblasHandle_t *h) { *h = (void *)2; return 0; }
static int hipblasDestroy(hipblasHandle_t h) { return 0; }
static int hipblasSetStream(hipblasHandle_t h, hipStream_t s) { return 0; }

/* device GEMMs: GEMMul8 bridge and native hipBLAS */
static int gemmul8_calls, native_calls, gemm_fault_call, gemm_fault_status;
static int device_gemm(int ta, int tb, int m, int n, int k, const double *alpha,
                       const double *a, int lda, const double *b, int ldb,
                       const double *beta, double *c, int ldc)
{
    if (gemmul8_calls + native_calls == gemm_fault_call) {
        for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) c[(size_t)j*ldc+i] = NAN;
        return gemm_fault_status;
    }
    dgemm_(ta == HIPBLAS_OP_N ? "N" : "T", tb == HIPBLAS_OP_N ? "N" : "T",
           &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
    return 0;
}
static hipblasStatus_t openmx_gemmul8Dgemm(hipblasHandle_t h, hipblasOperation_t ta, hipblasOperation_t tb,
    int m, int n, int k, const double *alpha, const double *a, int lda, const double *b, int ldb,
    const double *beta, double *c, int ldc)
{
    gemmul8_calls++;
    return device_gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
static hipblasStatus_t hipblasDgemm(hipblasHandle_t h, hipblasOperation_t ta, hipblasOperation_t tb,
    int m, int n, int k, const double *alpha, const double *a, int lda, const double *b, int ldb,
    const double *beta, double *c, int ldc)
{
    native_calls++;
    return device_gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
static int gemmul8_off;
static int openmx_gemmul8DgemmEnabled(void) { return !gemmul8_off; }
static size_t openmx_gemmul8DWorkspaceSize(int m, int n, int k)
{
    if (gemmul8_off) return 0;
    return (size_t)(15*((size_t)m*k + (size_t)k*n) + 64*(size_t)m*n + 1000);
}
static void openmx_gemmul8ReleaseWorkspaces(void) { event('R'); }

/* eigensolvers: MAGMA's two-stage solver and the LAPACK path of Krylov */
static int magma_calls, magma_fault, lapack_calls;
static void full_solve(int n, double *a, double *w)
{
    int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n, info;
    double *work = malloc((size_t)lwork*sizeof(double));
    int *iwork = malloc((size_t)liwork*sizeof(int));
    CHECK(work && iwork);
    dsyevd_("V", "L", &n, a, &n, w, work, &lwork, iwork, &liwork, &info);
    CHECK(info == 0);
    free(iwork); free(work);
}
static int openmx_magma_dsyevdx_gpu(int n, int maxn, double *d_a, double *w, int *mout)
{
    double all[64];
    magma_calls++;
    CHECK(n <= 64);
    if (magma_fault) { *mout = 0; for (int i = 0; i < n*n; i++) d_a[i] = NAN; return magma_fault; }
    full_solve(n, d_a, all);
    /* a sign flip the CPU solver would not make: rows of C are compared up to sign */
    for (int i = 0; i < n; i++) d_a[i] = -d_a[i];
    memcpy(w, all, (size_t)maxn*sizeof(double));
    *mout = maxn;
    return 0;
}
/* Eigen_lapack2: a[(i+1)*csize+j+1] in, eigenvector i at a[i*n+j] out, ko[1..EVmax] */
static void Eigen_lapack2(double *a, int csize, double *ko, int n, int EVmax)
{
    double *A = malloc(sizeof(double)*(size_t)n*n), w[64];
    lapack_calls++;
    CHECK(A && n <= 64);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) A[i*n+j] = a[(i+1)*csize+j+1];
    full_solve(n, A, w);
    for (int i = EVmax; i >= 1; i--) ko[i] = w[i-1];
    for (int i = 0; i < EVmax; i++) for (int j = 0; j < n; j++) a[i*n+j] = A[i*n+j];
    free(A);
}
'''


HARNESS = r'''
#define N 14
#define M3 8
#define MSIZE2 (N + 3)
#define CSIZE (N + 40)
static double worst;
static int checks, failures;
static double *Krylov_U_atoms[2], **Krylov_U_spin[1], ***Krylov_U;
static double **EC_atoms[2], ***EC_spin[1], ****EC_matrix;
static double H_input[N*N];
static int Msize[2] = {1, N}, Msize2[2] = {1, MSIZE2}, Msize3[2] = {1, M3};

static void setup(void)
{
    static double u[MSIZE2*M3 + 1], ec_rows[M3+1][M3+1], *ec[M3+1];
    for (int j = 0; j < N; j++) for (int i = 0; i <= j; i++) {
        double v = (i == j) ? -1.0 + 0.37*i : 0.21*sin(1.0 + i + 2.3*j);
        H_input[(size_t)j*N+i] = v; H_input[(size_t)i*N+j] = v;
    }
    u[0] = 1.0;   /* ZeroNum */
    for (int j = 0; j < M3; j++) for (int i = 0; i < MSIZE2; i++)
        u[j*MSIZE2+i+1] = cos(0.3 + 1.7*i + 0.9*j*j) + (i == j ? 1.0 : 0.0);
    for (int i = 0; i <= M3; i++) {
        ec[i] = ec_rows[i];
        for (int j = 0; j <= i; j++) ec_rows[i][j] = ec_rows[j][i] = 0.02*cos(0.5*i + 1.1*j) + (i == j ? 0.1*i : 0.0);
    }
    Krylov_U_atoms[1] = u; Krylov_U_spin[0] = Krylov_U_atoms; Krylov_U = Krylov_U_spin;
    EC_atoms[1] = ec; EC_spin[0] = EC_atoms; EC_matrix = EC_spin;
}

/* the sequence of Krylov_Col after the fused path declined (ws NULL: CPU) */
static void per_call(Krylov_GPU_Workspace *ws, double *H_DC, double *KU, double *C, double *ko)
{
    const int n = N, m3 = M3, msize2 = MSIZE2;
    for (int i = 0; i < n; i++) for (int j = 0; j < m3; j++) KU[j*n+i] = Krylov_U[0][1][j*msize2+i+1];
    Krylov_Dgemm(ws, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m3, n, H_DC, n, KU, n, C, n);
    Krylov_Dgemm(ws, HIPBLAS_OP_T, HIPBLAS_OP_N, m3, m3, n, KU, n, C, n, H_DC, m3);
    for (int i = 0; i < (int)Krylov_U[0][1][0]; i++) H_DC[i*m3+i] = 1.0e+3;
    for (int i = m3-1; 0 <= i; i--) for (int j = 0; j < m3; j++)
        H_DC[(i+1)*(m3+1)+(j+1)] = H_DC[i*m3+j] + EC_matrix[0][1][i+1][j+1];
    Krylov_Eigen2(ws, H_DC, m3+1, ko, m3, m3);
    Krylov_Dgemm(ws, HIPBLAS_OP_T, HIPBLAS_OP_T, m3, n, m3, H_DC, m3, KU, n, C, m3);
}

static void solve(Krylov_GPU_Workspace *ws, double *C, double *ko)
{
    static double H_DC[CSIZE*CSIZE], KU[(MSIZE2+2)*M3];
    memcpy(H_DC, H_input, sizeof(H_input));
    if (Krylov_ProjectedSolve_GPU(ws, 0, 1, Msize, Msize2, Msize3, Krylov_U, EC_matrix,
                                  H_DC, KU, C, ko) == 0)
        per_call(ws, H_DC, KU, C, ko);
}

static void compare(const double *C, const double *ko, const double *Cref, const double *koref)
{
    for (int k = 0; k < M3; k++) {
        double dot = 0.0, error = fabs(ko[k+1] - koref[k+1]);
        CHECK(isfinite(error) && error < 1e-10);
        if (worst < error) worst = error;
        for (int j = 0; j < N; j++) dot += C[k + (size_t)j*M3]*Cref[k + (size_t)j*M3];
        for (int j = 0; j < N; j++) {
            error = fabs(C[k + (size_t)j*M3] - (dot < 0 ? -1.0 : 1.0)*Cref[k + (size_t)j*M3]);
            CHECK(isfinite(error) && error < 1e-9);
            if (worst < error) worst = error;
        }
    }
    checks++;
}

static void reset(void)
{
    gemmul8_calls = native_calls = magma_calls = lapack_calls = 0;
    gemm_fault_call = gemm_fault_status = magma_fault = 0;
    Krylov_gpu_fits = 1;      /* what the preflight of the next call restores */
    events[0] = 0;
}

int main(void)
{
    static double Cref[CSIZE*CSIZE], koref[CSIZE], C[CSIZE*CSIZE], ko[CSIZE];
    static double H_DC[CSIZE*CSIZE], KU[(MSIZE2+2)*M3];
    Krylov_GPU_Workspace *ws;

    setup();
    memcpy(H_DC, H_input, sizeof(H_input));
    per_call(NULL, H_DC, KU, Cref, koref);
    CHECK(lapack_calls == 1);

    Krylov_GPU_InitOnce();
    Krylov_GPU_PreparePool(2);
    ws = Krylov_GPU_GetWorkspace(0);
    CHECK(ws != NULL && ws->active);
    Krylov_gpu_dgemm_min_flops = 0.0;

    for (int g8 = 1; g8 >= 0; g8--) {
        Krylov_gpu_gemmul8 = g8;
        for (int fused = 1; fused >= 0; fused--) {
            Krylov_gpu_fused = fused;
            for (int eig_min = 1; eig_min >= 0; eig_min--) {
                /* MAGMA from M3 on (eig_min 1) or LAPACK below 100 */
                Krylov_gpu_eigen_min = eig_min ? M3 : 100;
                reset();
                solve(ws, C, ko);
                compare(C, ko, Cref, koref);
                CHECK(gemmul8_calls == (g8 ? 3 : 0) && native_calls == (g8 ? 0 : 3));
                CHECK(magma_calls == (eig_min ? 1 : 0) && lapack_calls == (eig_min ? 0 : 1));
                CHECK(Krylov_gpu_fits);
                /* a failed MAGMA solve falls back to LAPACK */
                if (eig_min) {
                    reset(); magma_fault = 122;
                    solve(ws, C, ko);
                    compare(C, ko, Cref, koref);
                    CHECK(magma_calls == 1 && lapack_calls == 1 && Krylov_gpu_fits);
                }
                /* a device GEMM that fails is redone on the CPU, the rest of
                   the call stays there */
                for (int call = 1; call <= 3; call++) {
                    reset();
                    gemm_fault_call = call;
                    gemm_fault_status = (call == 2) ? HIPBLAS_STATUS_EXECUTION_FAILED : HIPBLAS_STATUS_ALLOC_FAILED;
                    solve(ws, C, ko);
                    compare(C, ko, Cref, koref);
                    CHECK(!Krylov_gpu_fits && gemmul8_calls + native_calls == call);
                    if (fused && call < 3)   /* the per-call path reran the atom on the CPU */
                        CHECK(magma_calls == 0 && lapack_calls == 1);
                    else if (eig_min)
                        CHECK(magma_calls == (call == 3 ? 1 : 0));
                    failures++;
                }
            }
        }
    }

    /* KU cache: built by the call's preparation, filled by the first solve,
       reused by the second */
    Krylov_gpu_gemmul8 = 1; Krylov_gpu_fused = 1; Krylov_gpu_eigen_min = M3;
    reset();
    Krylov_KUCache_Prepare(1, 1, Msize, Msize3, 1, 1, 0);
    CHECK(Krylov_kucache.state == 1 && Krylov_kucache.arena_elems == (size_t)N*M3);
    for (int pass = 0; pass < 2; pass++) {
        reset();
        solve(ws, C, ko);
        compare(C, ko, Cref, koref);
        CHECK(Krylov_kucache.valid[1] == 1);
    }

    /* preflight: the GEMMul8 workspace of the largest product on the GPU,
       once per thread of the pool, after the held workspaces are returned */
    {
        int msz[3] = {1, N, 2*N}, msz3[3] = {1, M3, M3 + 2};
        size_t per_thread = 0;
        unsigned long long need_on, need_off;
        for (int a = 1; a <= 2; a++) {
            int shape[3][3] = {{msz[a], msz3[a], msz[a]}, {msz3[a], msz3[a], msz[a]}, {msz3[a], msz[a], msz3[a]}};
            for (int s = 0; s < 3; s++) {
                size_t b = openmx_gemmul8DWorkspaceSize(shape[s][0], shape[s][1], shape[s][2]);
                if (per_thread < b) per_thread = b;
            }
        }
        Krylov_gpu_dgemm_min_flops = 0.0;
        Krylov_gpu_gemmul8 = 1; reset();
        Krylov_GPU_Preflight(2, msz, msz3);
        need_on = Krylov_gpu_need_total;
        CHECK(strcmp(events, "RBM") == 0 && Krylov_gpu_fits);
        Krylov_gpu_gemmul8 = 0; reset();
        Krylov_GPU_Preflight(2, msz, msz3);
        need_off = Krylov_gpu_need_total;
        CHECK(strcmp(events, "BM") == 0);
        CHECK(need_on - need_off == (unsigned long long)(per_thread*(size_t)Krylov_gpu_ws_pool_size));
        /* GEMMul8 off in the bridge (scf.gemmul8.enable off): no workspace */
        Krylov_gpu_gemmul8 = 1; gemmul8_off = 1; reset();
        Krylov_GPU_Preflight(2, msz, msz3);
        CHECK(Krylov_gpu_need_total == need_off && strcmp(events, "BM") == 0);
        gemmul8_off = 0;
        /* no product reaches the GPU: no workspace either */
        Krylov_gpu_dgemm_min_flops = 1e30; reset();
        Krylov_GPU_Preflight(2, msz, msz3);
        CHECK(Krylov_gpu_need_total == need_off && strcmp(events, "BM") == 0);
        /* the verdict counts the workspace: free memory between the two needs */
        Krylov_gpu_dgemm_min_flops = 0.0;
        free_memory = (size_t)((need_on + need_off)/2);
        reset();
        Krylov_GPU_Preflight(2, msz, msz3);
        CHECK(!Krylov_gpu_fits);
        Krylov_gpu_gemmul8 = 0; reset();
        Krylov_GPU_Preflight(2, msz, msz3);
        CHECK(Krylov_gpu_fits);
        free_memory = (size_t)8 << 30;
    }

    Krylov_Release_GPU_Caches();
    CHECK(device_allocations == 0);
    printf("PASS: %d numerical checks, %d injected GEMM failures; fused and per-call paths, "
           "GEMMul8/hipBLAS routing, eigensolver dispatch and MAGMA fallback, CPU redo of failed "
           "products, KU cache, preflight GEMMul8 workspace; max error %.3g\n", checks, failures, worst);
    return 0;
}
'''


def make_harness(source):
    macros = ["KRYLOV_ENABLE_GPU", "KRYLOV_GPU_EIGEN_MIN", "KRYLOV_GPU_DGEMM_MIN_FLOPS"]
    variables = ["Krylov_gpu_dgemm_min_flops", "Krylov_gpu_eigen_min", "Krylov_gpu_fused",
                 "Krylov_gpu_gemmul8", "Krylov_gpu_fits", "Krylov_gpu_alloc_failure_reported",
                 "Krylov_gpu_device_ranks", "Krylov_gpu_need_total"]
    parts = [PREFIX, *(define(source, name) for name in macros),
             *(variable(source, name) for name in variables),
             function(source, "Krylov_GPU_Enabled"),
             typedef(source, "Krylov_GPU_Workspace"),
             between(source, "static void Krylov_GPU_InitOnce(void);",
                     "static void Krylov_Eigen2(Krylov_GPU_Workspace *ws, double *a, int csize, "
                     "double *ko, int n, int EVmax);"),
             variable(source, "Krylov_gpu_ws_pool"), variable(source, "Krylov_gpu_ws_pool_size")]
    names = ["Krylov_GPU_InitOnce", "Krylov_GPU_PreparePool", "Krylov_GPU_GetWorkspace",
             "Krylov_GPU_ReportAllocFailure", "Krylov_GPU_EnsureDouble", "Krylov_GPU_Usable",
             "Krylov_GPU_EnsureHostMatrix", "Krylov_GPU_Workspace_Init", "Krylov_GPU_Workspace_Free",
             "Krylov_GPU_EnsureHipblas", "Krylov_GPU_DeviceDgemm", "Krylov_GPU_GemmFailed",
             "Krylov_Dgemm", "Krylov_Eigen2"]
    parts += [function(source, name) for name in names]
    parts += [typedef(source, "Krylov_KUCache"), variable(source, "Krylov_kucache"),
              function(source, "Krylov_Release_GPU_KUCache", static=False),
              function(source, "Krylov_Release_GPU_Caches", static=False)]
    names = ["Krylov_KUCache_Prepare", "Krylov_GPU_Preflight", "Krylov_KUCache_Get",
             "Krylov_ProjectedSolve_GPU"]
    parts += [function(source, name) for name in names]
    return "\n\n".join(parts + [HARNESS])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--libs", default=os.environ.get("LAPACK_LIBS", "-llapack -lblas"),
                        help="LP64 CPU LAPACK/BLAS libraries (default: -llapack -lblas)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / "source/Krylov.c").read_text()
    flags = shlex.split(os.environ.get("CFLAGS", "-O2"))
    if args.sanitize:
        flags += ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix="openmx-krylov-gpu-") as directory:
        harness = Path(directory) / "krylov_gpu.c"
        binary = Path(directory) / "krylov_gpu"
        harness.write_text(make_harness(source))
        subprocess.run([*shlex.split(args.cc), *flags, "-std=c11", str(harness),
                        *shlex.split(args.libs), "-lm", "-o", str(binary)], check=True)
        env = os.environ.copy()
        for name in ("OPENMX_KRYLOV_GPU", "OPENMX_KRYLOV_GEMMUL8", "OPENMX_KRYLOV_GPU_FUSED",
                     "OPENMX_KRYLOV_GPU_EIGEN_MIN", "OPENMX_KRYLOV_GPU_DGEMM_MIN_FLOPS",
                     "OPENMX_KRYLOV_GPU_KU_CACHE", "OPENMX_KRYLOV_GPU_KU_RESERVE_MB",
                     "OPENMX_KRYLOV_GPU_RESERVE_MB", "OPENMX_KRYLOV_GPU_PROFILE"):
            env.pop(name, None)
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", BLIS_NUM_THREADS="1")
        subprocess.run([str(binary)], env=env, check=True)


if __name__ == "__main__":
    main()
