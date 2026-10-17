#!/usr/bin/env python3
"""Check the noncollinear DC-LNO GPU solve and its failure handling without
a GPU or MPI (the sibling of run_dclno_solver_retry.py).

The production helpers of source/Divide_Conquer_LNO.c (DCLNO_GpuSolverZ_*,
DCLNO_NonColGpuCallFailed, DCLNO_Solve_NonCol_GpuSolver) are extracted
verbatim and built against host stubs of the HIP/hipBLAS/GEMMul8 calls and
of openmx_hipsolver_zheevd_gpu.  The solver stub is LAPACK zheevd whose
vectors get a random complex phase per column, like the phase similarity of
the real wrapper, so the back transformation with conj(T) is exercised.

Checked:
  * success: the eigenvalues and the projector sum_k conj(x_k) x_k^T of the
    returned states (Hmat(state, basis)) against LAPACK zhegv for conj(H) and
    S (the problem the CPU flow of DC_NonCol solves), for a non-collinear
    Hamiltonian with complex spin off-diagonal blocks, a real block-diagonal
    one and a non-magnetic one (exactly degenerate pairs);
  * injected failures of the eigensolver in the overlap (stage 1) and the
    Hamiltonian (stage 2) solve:
      1 INFO > 0                         -> CPU retry (status INFO)
      2 MAGMA_ERR_DEVICE_ALLOC (-113)    -> CPU retry (no memory)
      3 MAGMA_ERR_HOST_ALLOC (-112)      -> CPU retry (no memory)
      4 MAGMA_ERR_UNKNOWN (-116): NaN    -> CPU retry (GPU error)
      5 hipSOLVER status (positive)      -> CPU retry (status)
      6 INFO 0 with a NaN eigenvalue     -> CPU retry (GPU error)
      7 MAGMA_ERR_ILLEGAL_VALUE (-104)   -> fatal
      8 a wrong eigenpair count          -> fatal
      9 another MAGMA error (-118, from the wrapper's MAGMA retry)
                                         -> CPU retry (GPU error)
     10 a LAPACK argument error (-7)     -> fatal
  * failures of the five hipBLAS/GEMMul8 calls (allocation failure -> no
    memory, other status -> GPU error), of hipMalloc and of hipblasCreate;
  * the host inputs survive every failure bit for bit (the CPU flow of
    DC_NonCol retries on them), a later solve does not inherit a failed
    one's device state, and every device buffer is released at the end;
  * OPENMX_DCLNO_RETRY_TEST=1 turns a successful overlap solve into a retry.
This checks the control flow and the numbers, not rocSOLVER or GPU
execution.  --sanitize adds ASan/UBSan.
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


def define(source, name):
    match = re.search(r"^#define\s+" + re.escape(name) + r"\b.*$", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production macro {name}")
    return match.group(0)


PREFIX = r'''
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
static int level_stdout=3;
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); \
} } while (0)
typedef struct { double r, i; } dcomplex;
typedef struct { double x, y; } hipDoubleComplex;
#define make_hipDoubleComplex(a, b) ((hipDoubleComplex){(a), (b)})
typedef void *hipStream_t;
typedef void *hipblasHandle_t;
enum { hipSuccess, hipStreamNonBlocking, hipMemcpyHostToDevice, hipMemcpyDeviceToHost,
       hipMemcpyDeviceToDevice };
enum { HIPBLAS_STATUS_SUCCESS = 0, HIPBLAS_STATUS_ALLOC_FAILED = 2,
       HIPBLAS_STATUS_EXECUTION_FAILED = 6 };
enum { HIPBLAS_SIDE_RIGHT = 142, HIPBLAS_OP_N = 111, HIPBLAS_OP_T = 112, HIPBLAS_OP_C = 113 };
typedef struct { double re, im; } zc;   /* Fortran COMPLEX*16 */
extern void zheevd_(const char *, const char *, const int *, zc *, const int *, double *,
                    zc *, const int *, double *, const int *, int *, const int *, int *);
extern void zgemm_(const char *, const char *, const int *, const int *, const int *,
                   const zc *, const zc *, const int *, const zc *, const int *,
                   const zc *, zc *, const int *);
extern void zhegv_(const int *, const char *, const char *, const int *, zc *, const int *,
                   zc *, const int *, double *, zc *, const int *, double *, int *);

static jmp_buf fatal_jump;
static int fatal_expected, fatal_seen;
static int solver_calls, fault_stage, fault_kind;
static int blas_calls, blas_fault_call, blas_fault_status;
static int malloc_calls, malloc_fault_call, create_fault;
static int device_allocations, phase_mode;
static unsigned phase_seed = 12345u;
static void DCLNO_AbortWithMessage(const char *message)
{
    if (!fatal_expected) { fprintf(stderr, "Unexpected fatal: %s\n", message); exit(3); }
    fatal_seen++;
    longjmp(fatal_jump, 1);
}
#define wait_hipfunc(call) CHECK((call) == 0)
static int hipStreamCreateWithFlags(hipStream_t *p, int flags) { *p = (void *)1; return 0; }
static int hipStreamDestroy(hipStream_t s) { return 0; }
static int hipStreamSynchronize(hipStream_t s) { return 0; }
static int hipDeviceSynchronize(void) { return 0; }
static int hipGetLastError(void) { return 0; }
static int hipblasCreate(hipblasHandle_t *p)
{
    if (create_fault) { *p = NULL; return HIPBLAS_STATUS_ALLOC_FAILED; }
    *p = (void *)2; return 0;
}
static int hipblasSetStream(hipblasHandle_t h, hipStream_t s) { return 0; }
static int hipblasDestroy(hipblasHandle_t h) { return 0; }
static int hipMalloc(void **p, size_t bytes)
{
    if (++malloc_calls == malloc_fault_call) { *p = (void *)0x1; return 2; }  /* garbage pointer */
    *p = malloc(bytes ? bytes : 1); CHECK(*p != NULL); device_allocations++; return 0;
}
static int hipFree(void *p) { if (p) { free(p); device_allocations--; } return 0; }
static int hipMemcpyAsync(void *to, const void *from, size_t bytes, int kind, hipStream_t s)
{
    memcpy(to, from, bytes); return 0;
}
static int hipMemsetAsync(void *to, int value, size_t bytes, hipStream_t s)
{
    memset(to, value, bytes); return 0;
}
static int hipMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                            size_t width, size_t height, int kind, hipStream_t s)
{
    for (size_t r = 0; r < height; r++)
        memcpy((char *)dst + r * dpitch, (const char *)src + r * spitch, width);
    return 0;
}
static int blas_fault(void)
{
    if (++blas_calls == blas_fault_call) return blas_fault_status;
    return 0;
}
/* C = A diag(x) */
static int hipblasZdgmm(hipblasHandle_t h, int side, int m, int n, const hipDoubleComplex *a, int lda,
                        const hipDoubleComplex *x, int incx, hipDoubleComplex *c, int ldc)
{
    int status = blas_fault();
    if (status) return status;
    CHECK(side == HIPBLAS_SIDE_RIGHT);
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
        hipDoubleComplex u = a[(size_t)j*lda+i], v = x[(size_t)j*incx];
        c[(size_t)j*ldc+i].x = u.x*v.x - u.y*v.y;
        c[(size_t)j*ldc+i].y = u.x*v.y + u.y*v.x;
    }
    return 0;
}
static const char *op_name(int op) { return op == HIPBLAS_OP_N ? "N" : op == HIPBLAS_OP_T ? "T" : "C"; }
static int openmx_gemmul8Zgemm(hipblasHandle_t h, int op_a, int op_b, int m, int n, int k,
                               const hipDoubleComplex *alpha, const hipDoubleComplex *a, int lda,
                               const hipDoubleComplex *b, int ldb, const hipDoubleComplex *beta,
                               hipDoubleComplex *c, int ldc)
{
    int status = blas_fault();
    if (status) return status;
    zgemm_(op_name(op_a), op_name(op_b), &m, &n, &k, (const zc *)alpha, (const zc *)a, &lda,
           (const zc *)b, &ldb, (const zc *)beta, (zc *)c, &ldc);
    return 0;
}
/* C = alpha op(A) + beta op(B); DC-LNO uses op(A) = A^T with beta = 0 */
static int hipblasZgeam(hipblasHandle_t h, int op_a, int op_b, int m, int n,
                        const hipDoubleComplex *alpha, const hipDoubleComplex *a, int lda,
                        const hipDoubleComplex *beta, const hipDoubleComplex *b, int ldb,
                        hipDoubleComplex *c, int ldc)
{
    int status = blas_fault();
    if (status) return status;
    CHECK(op_a == HIPBLAS_OP_T && beta->x == 0.0 && beta->y == 0.0 && alpha->x == 1.0 && alpha->y == 0.0);
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) c[(size_t)j*ldc+i] = a[(size_t)i*lda+j];
    return 0;
}
static double phase_random(void)
{
    phase_seed = phase_seed * 1664525u + 1013904223u;
    return (double)(phase_seed >> 8) / (double)(1u << 24);
}
/* rocSOLVER zheevd behind the phase similarity: all n vectors replace a,
   the lowest maxn eigenvalues go to w; the vectors carry an arbitrary
   complex phase per column (phase_mode 1) */
static int openmx_hipsolver_zheevd_gpu(int n, int maxn, void *d_a, double *w, int *mout)
{
    zc *a = d_a;
    solver_calls++;
    *mout = 0;
    if (solver_calls == fault_stage && fault_kind && fault_kind != 6 && fault_kind != 8) {
        /* failed device solves may destroy their input */
        for (int i = 0; i < n * n; i++) a[i].re = a[i].im = NAN;
        for (int i = 0; i < maxn; i++) w[i] = NAN;
        switch (fault_kind) {
        case 1: return 122;
        case 2: return -113;
        case 3: return -112;
        case 4: return -116;
        case 5: return 2;
        case 9: return -118;
        case 10: return -7;
        default: return -104;
        }
    }
    int lwork = 2*n + n*n, lrwork = 1 + 5*n + 2*n*n, liwork = 3 + 5*n, info;
    zc *work = malloc((size_t)lwork * sizeof(*work));
    double *rwork = malloc((size_t)lrwork * sizeof(*rwork)), *all = malloc((size_t)n * sizeof(*all));
    int *iwork = malloc((size_t)liwork * sizeof(*iwork));
    CHECK(work && rwork && all && iwork);
    zheevd_("V", "L", &n, a, &n, all, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
    CHECK(info == 0);
    if (phase_mode) {
        for (int j = 0; j < n; j++) {
            double t = 6.283185307179586 * phase_random(), c = cos(t), s = sin(t);
            for (int i = 0; i < n; i++) {
                zc v = a[(size_t)j*n+i];
                a[(size_t)j*n+i].re = c*v.re - s*v.im;
                a[(size_t)j*n+i].im = c*v.im + s*v.re;
            }
        }
    }
    memcpy(w, all, (size_t)maxn * sizeof(double));
    free(iwork); free(all); free(rwork); free(work);
    *mout = maxn;
    if (solver_calls == fault_stage && fault_kind == 6) w[maxn > 1 ? maxn/2 : 0] = NAN;
    if (solver_calls == fault_stage && fault_kind == 8) *mout = maxn - 1;
    return 0;
}
static int openmx_hipsolver_release_workspace(void) { return 0; }
'''


HARNESS = r'''
#define MAX_NH 5
#define MAX_N (2*MAX_NH)
static double worst_error;
static int numerical_checks, failure_checks;
static const double sentinel = -12345.25;

/* S: NUMH x NUMH real (complex storage), H: NUM x NUM Hermitian, column-major */
static void matrices(int nh, int kind, dcomplex *s, dcomplex *h)
{
    const int n = 2*nh;
    for (int j = 0; j < nh; j++) for (int i = 0; i < nh; i++) {
        s[(size_t)j*nh+i].r = i == j ? 1.4 + 0.1*i : 0.05*cos(1.0 + i + 2.0*j) * 0.5 + 0.05*cos(1.0 + j + 2.0*i) * 0.5;
        s[(size_t)j*nh+i].i = 0.0;
    }
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { h[(size_t)j*n+i].r = 0.0; h[(size_t)j*n+i].i = 0.0; }
    for (int b = 0; b < 2; b++)
        for (int j = 0; j < nh; j++) for (int i = 0; i < nh; i++) {
            /* the spin-diagonal blocks are real symmetric; kind 2 makes them equal */
            double shift = (kind == 2) ? 0.0 : 0.13*b;
            double v = (i == j) ? -1.0 + 0.45*i + shift
                                : 0.11*sin(2.0 + i + j + ((kind == 2) ? 0.0 : 0.7*b));
            h[(size_t)(j + b*nh)*n + i + b*nh].r = v;
        }
    if (kind == 0)   /* non-collinear: complex spin off-diagonal block */
        for (int j = 0; j < nh; j++) for (int i = 0; i < nh; i++) {
            double re = 0.04*cos(3.0 + i - 2.0*j), im = 0.05*sin(1.0 + 2.0*i + j);
            h[(size_t)(j + nh)*n + i].r = re;  h[(size_t)(j + nh)*n + i].i = im;
            h[(size_t)i*n + j + nh].r = re;    h[(size_t)i*n + j + nh].i = -im;
        }
}

/* the problem the CPU flow of DC_NonCol solves: conj(H) x = lambda S2 x */
static void reference(int nh, const dcomplex *s, const dcomplex *h, double *w, zc *x)
{
    const int n = 2*nh, itype = 1, lwork = 4*MAX_N;
    zc b[MAX_N*MAX_N], work[4*MAX_N];
    double rwork[3*MAX_N];
    int info;
    memset(b, 0, sizeof(b));
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) {
        x[(size_t)j*n+i].re = h[(size_t)j*n+i].r;
        x[(size_t)j*n+i].im = -h[(size_t)j*n+i].i;
    }
    for (int bk = 0; bk < 2; bk++)
        for (int j = 0; j < nh; j++) for (int i = 0; i < nh; i++)
            b[(size_t)(j + bk*nh)*n + i + bk*nh].re = s[(size_t)j*nh+i].r;
    zhegv_(&itype, "V", "L", &n, x, &n, b, &n, w, work, &lwork, rwork, &info);
    CHECK(info == 0);
}

/* projector over the lowest m states, P(a,b) = sum_k conj(x_k(a)) x_k(b) */
static void projector(int n, int m, const zc *x, int state_major, zc *p)
{
    for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) {
        double re = 0.0, im = 0.0;
        for (int k = 0; k < m; k++) {
            zc u = state_major ? x[(size_t)a*n + k] : x[(size_t)k*n + a];
            zc v = state_major ? x[(size_t)b*n + k] : x[(size_t)k*n + b];
            re += u.re*v.re + u.im*v.im;
            im += u.re*v.im - u.im*v.re;
        }
        p[(size_t)b*n+a].re = re; p[(size_t)b*n+a].im = im;
    }
}

static void validate(int nh, int kind, int k, const dcomplex *s, const dcomplex *h,
                     const double *ko, const dcomplex *hout)
{
    const int n = 2*nh;
    /* the non-magnetic Hamiltonian has degenerate pairs: compare whole pairs */
    const int m = (kind == 2) ? k - k % 2 : k;
    double w[MAX_N];
    zc x[MAX_N*MAX_N], p_ref[MAX_N*MAX_N], p[MAX_N*MAX_N];
    reference(nh, s, h, w, x);
    for (int q = 0; q < k; q++) {
        double error = fabs(ko[q+1] - w[q]);
        CHECK(isfinite(error) && error < 1e-12);
        if (worst_error < error) worst_error = error;
    }
    if (m > 0) {
        projector(n, m, x, 0, p_ref);
        projector(n, m, (const zc *)hout, 1, p);
        for (int i = 0; i < n*n; i++) {
            double error = hypot(p[i].re - p_ref[i].re, p[i].im - p_ref[i].im);
            CHECK(isfinite(error) && error < 1e-12);
            if (worst_error < error) worst_error = error;
        }
    }
    numerical_checks++;
}

static int retried(int kind) { return (kind >= 1 && kind <= 6) || kind == 9; }

static int expected_status(int kind)
{
    switch (kind) {
    case 1: return 122;
    case 2: case 3: return DCLNO_RETRY_NO_MEMORY;
    case 4: case 6: case 9: return DCLNO_RETRY_GPU_ERROR;
    case 5: return 2;
    default: return -1;
    }
}

static void reset_faults(void)
{
    solver_calls = 0; fault_stage = 0; fault_kind = 0;
    blas_calls = 0; blas_fault_call = 0; blas_fault_status = 0;
    malloc_calls = 0; malloc_fault_call = 0; create_fault = 0;
}

/* one call; stage/kind inject an eigensolver failure, blas_call/blas_status
   a hipBLAS/GEMMul8 failure */
static void run_case(int nh, int kind, int k, int stage, int fault, int blas_call, int blas_status)
{
    const int n = 2*nh;
    dcomplex s[MAX_NH*MAX_NH], h[MAX_N*MAX_N], s0[MAX_NH*MAX_NH], h0[MAX_N*MAX_N];
    double ko[MAX_N+2];
    int info, expect_retry;
    matrices(nh, kind, s, h);
    memcpy(s0, s, sizeof(dcomplex)*(size_t)nh*nh);
    memcpy(h0, h, sizeof(dcomplex)*(size_t)n*n);
    for (int i = 0; i < MAX_N+2; i++) ko[i] = sentinel;
    reset_faults();
    fault_stage = stage; fault_kind = fault;
    blas_fault_call = blas_call; blas_fault_status = blas_status;

    if (fault && !retried(fault)) {
        fatal_expected = 1; fatal_seen = 0;
        if (setjmp(fatal_jump) == 0) {
            (void)DCLNO_Solve_NonCol_GpuSolver(n, nh, k, 1, s, h, ko);
            CHECK(0 && "an invalid eigensolver result was accepted");
        }
        fatal_expected = 0;
        CHECK(fatal_seen == 1 && solver_calls == stage);
        CHECK(memcmp(s, s0, sizeof(dcomplex)*(size_t)nh*nh) == 0);
        CHECK(memcmp(h, h0, sizeof(dcomplex)*(size_t)n*n) == 0);
        failure_checks++;
        return;
    }
    info = DCLNO_Solve_NonCol_GpuSolver(n, nh, k, 1, s, h, ko);
    CHECK(memcmp(s, s0, sizeof(dcomplex)*(size_t)nh*nh) == 0);
    expect_retry = fault || blas_call;
    if (expect_retry) {
        if (fault) CHECK(info == expected_status(fault) && solver_calls == stage);
        else CHECK(info == (blas_status == HIPBLAS_STATUS_ALLOC_FAILED ? DCLNO_RETRY_NO_MEMORY
                                                                       : DCLNO_RETRY_GPU_ERROR));
        /* the CPU flow of DC_NonCol solves again from the untouched inputs */
        CHECK(memcmp(h, h0, sizeof(dcomplex)*(size_t)n*n) == 0);
        failure_checks++;
    } else {
        CHECK(info == 0 && solver_calls == 2 && blas_calls == 5);
        validate(nh, kind, k, s0, h0, ko, h);
    }
    CHECK(ko[n+1] == sentinel);
}

static void allocation_failures(int nh, int kind)
{
    const int n = 2*nh;
    dcomplex s[MAX_NH*MAX_NH], h[MAX_N*MAX_N], h0[MAX_N*MAX_N];
    double ko[MAX_N+2];
    for (int which = 1; which <= 5; which++) {
        /* which 1-4: the four device buffers; 5: the hipBLAS handle */
        DCLNO_GpuSolverZ_Destroy();
        CHECK(device_allocations == 0);
        matrices(nh, kind, s, h);
        memcpy(h0, h, sizeof(h0));
        reset_faults();
        if (which <= 4) malloc_fault_call = which; else create_fault = 1;
        CHECK(DCLNO_Solve_NonCol_GpuSolver(n, nh, n, 1, s, h, ko) == DCLNO_RETRY_NO_MEMORY);
        CHECK(memcmp(h, h0, sizeof(dcomplex)*(size_t)n*n) == 0);
        CHECK(device_allocations == 0 && solver_calls == 0);
        failure_checks++;
    }
    reset_faults();
}

int main(int argc, char **argv)
{
    const int sizes[] = {2, 5};
    if (argc > 1 && strcmp(argv[1], "retry-test") == 0) {
        /* OPENMX_DCLNO_RETRY_TEST=1: a successful overlap solve reports INFO 1 */
        dcomplex s[MAX_NH*MAX_NH], h[MAX_N*MAX_N], h0[MAX_N*MAX_N];
        double ko[MAX_N+2];
        matrices(5, 0, s, h);
        memcpy(h0, h, sizeof(h0));
        reset_faults();
        CHECK(DCLNO_Solve_NonCol_GpuSolver(10, 5, 10, 1, s, h, ko) == 1 && solver_calls == 1);
        CHECK(memcmp(h, h0, sizeof(dcomplex)*100) == 0);
        DCLNO_GpuSolverZ_Destroy();
        CHECK(device_allocations == 0);
        printf("PASS: OPENMX_DCLNO_RETRY_TEST turns a successful overlap solve into a CPU retry\n");
        return 0;
    }
    for (phase_mode = 0; phase_mode <= 1; phase_mode++)
    for (int a = 0; a < 2; a++) {
        const int nh = sizes[a], n = 2*nh, counts[] = {1, n-1, n};
        for (int kind = 0; kind < 3; kind++) {
            for (int b = 0; b < 3; b++) {
                const int k = counts[b];
                run_case(nh, kind, k, 0, 0, 0, 0);
                for (int stage = 1; stage <= 2; stage++) {
                    for (int fault = 1; fault <= 10; fault++) {
                        run_case(nh, kind, k, stage, fault, 0, 0);
                        /* a later problem must not inherit a failed solve's
                           poisoned device matrices */
                        run_case(nh, kind, k, 0, 0, 0, 0);
                    }
                }
                for (int call = 1; call <= 5; call++) {
                    run_case(nh, kind, k, 0, 0, call, HIPBLAS_STATUS_ALLOC_FAILED);
                    run_case(nh, kind, k, 0, 0, call, HIPBLAS_STATUS_EXECUTION_FAILED);
                    run_case(nh, kind, k, 0, 0, 0, 0);
                }
            }
            allocation_failures(nh, kind);
            run_case(nh, kind, n, 0, 0, 0, 0);
        }
    }
    DCLNO_GpuSolverZ_ReleaseDeviceMemory();
    CHECK(device_allocations == 0);
    run_case(5, 0, 10, 0, 0, 0, 0);
    DCLNO_GpuSolverZ_Destroy();
    CHECK(device_allocations == 0);
    printf("PASS: %d injected failures, %d numerical checks; eigensolver INFO/status, no memory, "
           "non-finite results, rejected arguments, eigenpair count, hipBLAS/GEMMul8 and allocation "
           "failures, original inputs, recovery, conj(T) back transformation with phased vectors; "
           "max error %.3g\n", failure_checks, numerical_checks, worst_error);
    return 0;
}
'''


def make_harness(source):
    context_end = source.index("} DCLNO_GpuSolverZCtx;") + len("} DCLNO_GpuSolverZCtx;")
    context_start = source.rfind("typedef struct {", 0, context_end)
    macros = ["DCLNO_MAGMA_ERR_HOST_ALLOC", "DCLNO_MAGMA_ERR_DEVICE_ALLOC", "DCLNO_RETRY_NO_MEMORY",
              "DCLNO_RETRY_GPU_ERROR", "DCLNO_MAGMA_ERR", "DCLNO_MAGMA_ERR_ILLEGAL_VALUE",
              "DCLNO_MAGMA_ERR_UNKNOWN"]
    names = ["DCLNO_CheckedArrayBytes", "DCLNO_CheckedMulCount", "DCLNO_MallocArray",
             "DCLNO_RetryTest", "DCLNO_ReportGpuRetry"]
    parts = [PREFIX, *(define(source, name) for name in macros),
             *(function(source, name) for name in names),
             source[context_start:context_end],
             "static DCLNO_GpuSolverZCtx DCLNO_gpusolverz_ctx = {0};"]
    names = ["DCLNO_GpuSolverZ_FreeBuffers", "DCLNO_GpuSolverZ_Destroy", "DCLNO_GpuSolverZ_Init",
             "DCLNO_GpuSolverZ_EnsureMatrixCapacity", "DCLNO_GpuSolverZ_Eigen",
             "DCLNO_NonColGpuCallFailed", "DCLNO_Solve_NonCol_GpuSolver",
             "DCLNO_GpuSolverZ_ReleaseDeviceMemory"]
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
    source = (root / "source/Divide_Conquer_LNO.c").read_text()
    flags = shlex.split(os.environ.get("CFLAGS", "-O2"))
    if args.sanitize:
        flags += ["-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix="openmx-dclno-nc-retry-") as directory:
        harness = Path(directory) / "nc_retry.c"
        binary = Path(directory) / "nc_retry"
        harness.write_text(make_harness(source))
        subprocess.run([*shlex.split(args.cc), *flags, "-std=c11", str(harness),
                        *shlex.split(args.libs), "-lm", "-o", str(binary)], check=True)
        env = os.environ.copy()
        env.pop("OPENMX_DCLNO_RETRY_TEST", None)
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", BLIS_NUM_THREADS="1")
        subprocess.run([str(binary)], env=env, check=True)
        env["OPENMX_DCLNO_RETRY_TEST"] = "1"
        subprocess.run([str(binary), "retry-test"], env=env, check=True)


if __name__ == "__main__":
    main()
