#!/usr/bin/env python3
"""Check the DC-LNO local-solve failure handling without a GPU or MPI.

The production helpers of source/Divide_Conquer_LNO.c are extracted
verbatim and built against host stubs of the HIP/hipBLAS calls and of
openmx_magma_dsyevdx_gpu (the AMD port of the NVIDIA test of the same name,
whose stubs are cuSOLVER's).  The stubs inject MAGMA results:
  1 INFO > 0 (no convergence)            -> CPU retry
  2 INFO < 0 (rejected argument)         -> fatal
  3 a wrong eigenpair count              -> fatal
  4 MAGMA_ERR_DEVICE_ALLOC (-113)        -> CPU retry (no device memory)
  5 MAGMA_ERR_HOST_ALLOC (-112)          -> CPU retry
  6 another MAGMA error (-116, e.g. a failed magma_init) -> fatal, reported
    as a MAGMA error, not as a rejected argument
  7 MAGMA_ERR_ILLEGAL_VALUE (-104)       -> fatal (rejected argument)
in the overlap (stage 1) and the Hamiltonian (stage 2) solve.  Successful
solves and the CPU retry use real LAPACK/BLAS; the results are checked
against LAPACK dsygv, and the host inputs must survive every failure.
This checks the control flow, the preservation of the inputs and the
numbers, not MAGMA or GPU execution.  --sanitize adds ASan/UBSan.
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
typedef int INTEGER;
typedef void *hipStream_t;
typedef void *hipblasHandle_t;
enum { hipSuccess, hipStreamNonBlocking, hipMemcpyHostToDevice, hipMemcpyDeviceToHost,
       HIPBLAS_SIDE_RIGHT, HIPBLAS_OP_N, HIPBLAS_OP_T, HIPBLAS_OP_C };
#define F77_NAME(lower, upper) lower ## _
extern void dsyevd_(const char *, const char *, const int *, double *, const int *,
                    double *, double *, const int *, int *, const int *, int *);
extern void dgemm_(const char *, const char *, const int *, const int *, const int *,
                   const double *, const double *, const int *, const double *,
                   const int *, const double *, double *, const int *);
extern void dsygv_(const int *, const char *, const char *, const int *, double *,
                   const int *, double *, const int *, double *, double *,
                   const int *, int *);

static jmp_buf fatal_jump;
static int fatal_expected, fatal_seen;
static char fatal_message[512];
static int fault_stage, fault_kind, solver_calls, device_allocations;
static const double sentinel = -12345.25;
static void DCLNO_AbortWithMessage(const char *message)
{
    if (!fatal_expected) { fprintf(stderr, "Unexpected fatal: %s\n", message); exit(3); }
    snprintf(fatal_message, sizeof(fatal_message), "%s", message);
    fatal_seen++;
    longjmp(fatal_jump, 1);
}
#define wait_hipfunc(call) CHECK((call) == 0)
static int hipStreamCreateWithFlags(hipStream_t *p, int flags) { *p = (void *)1; return 0; }
static int hipblasCreate(hipblasHandle_t *p) { *p = (void *)2; return 0; }
static int hipblasSetStream(hipblasHandle_t h, hipStream_t s) { return 0; }
static int hipStreamSynchronize(hipStream_t s) { return 0; }
static int hipStreamDestroy(hipStream_t s) { return 0; }
static int hipblasDestroy(hipblasHandle_t h) { return 0; }
static int hipMalloc(void **p, size_t bytes)
{
    *p = malloc(bytes ? bytes : 1); CHECK(*p != NULL); device_allocations++; return 0;
}
static int hipFree(void *p) { if (p) { free(p); device_allocations--; } return 0; }
static int hipMemcpyAsync(void *to, const void *from, size_t bytes, int kind, hipStream_t stream)
{
    memcpy(to, from, bytes); return 0;
}
/* MAGMA's two-stage solver: the first maxn eigenpairs of the lower
   triangle, the vectors in a (lda = n) */
static int openmx_magma_dsyevdx_gpu(int n, int maxn, double *a, double *w, int *mout)
{
    solver_calls++;
    *mout = maxn;
    if (solver_calls == fault_stage && fault_kind) {
        /* failed device solves may destroy their input and output */
        for (int i = 0; i < n * n; i++) a[i] = NAN;
        for (int i = 0; i < n; i++) w[i] = NAN;
        switch (fault_kind) {
        case 1: return 122;
        case 2: return -7;
        case 3: *mout = maxn - 1; return 0;
        case 4: return -113;
        case 5: return -112;
        case 6: return -116;
        default: return -104;
        }
    }
    int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n, result;
    double *work = malloc((size_t)lwork * sizeof(*work));
    double *all = malloc((size_t)n * sizeof(*all));
    int *iwork = malloc((size_t)liwork * sizeof(*iwork));
    CHECK(work && iwork && all);
    dsyevd_("V", "L", &n, a, &n, all, work, &lwork, iwork, &liwork, &result);
    memcpy(w, all, (size_t)maxn * sizeof(double));
    free(iwork); free(all); free(work);
    return result;
}
static int hipblasDdgmm(hipblasHandle_t handle, int side, int m, int n,
    const double *a, int lda, const double *x, int incx, double *c, int ldc)
{
    CHECK(side == HIPBLAS_SIDE_RIGHT);
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++)
        c[(size_t)j*ldc+i] = a[(size_t)j*lda+i] * x[j*incx];
    return 0;
}
static int openmx_gemmul8Dgemm(hipblasHandle_t handle, int op_a, int op_b,
    int m, int n, int k, const double *alpha, const double *a, int lda,
    const double *b, int ldb, const double *beta, double *c, int ldc)
{
    const char *ta = op_a == HIPBLAS_OP_N ? "N" : "T";
    const char *tb = op_b == HIPBLAS_OP_N ? "N" : "T";
    dgemm_(ta, tb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
    return 0;
}
'''


HARNESS = r'''
#define MAX_N 7
static double worst_error;
static int numerical_checks, failure_checks;

static void matrices(int n, double *s, double *h)
{
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) {
        s[(size_t)j*n+i] = i == j ? 1.5 + 0.2*i : 0.03*cos(i+j+1.0);
        h[(size_t)j*n+i] = i == j ? -2.0 + 0.7*i : 0.2*sin(i+j+2.0);
    }
}

static void reference(int n, const double *s, const double *h, double *w)
{
    double a[MAX_N*MAX_N], b[MAX_N*MAX_N], work[8*MAX_N];
    int itype = 1, lwork = 8*MAX_N, info;
    memcpy(a, h, (size_t)n*n*sizeof(double));
    memcpy(b, s, (size_t)n*n*sizeof(double));
    dsygv_(&itype, "V", "L", &n, a, &n, b, &n, w, work, &lwork, &info);
    CHECK(info == 0);
}

static void validate(int n, int k, const double *s, const double *h,
                     const double *ko, double **c)
{
    double expected[MAX_N], scale_h = 0, scale_s = 0;
    reference(n, s, h, expected);
    for (int i = 0; i < n*n; i++) { scale_h += h[i]*h[i]; scale_s += s[i]*s[i]; }
    scale_h = sqrt(scale_h); scale_s = sqrt(scale_s);
    for (int q = 0; q < k; q++) {
        double residual = 0, norm = 0;
        double error = fabs(ko[q+1] - expected[q]);
        CHECK(isfinite(error) && error < 1e-12);
        if (worst_error < error) worst_error = error;
        for (int i = 0; i < n; i++) {
            double hv = 0, sv = 0;
            for (int j = 0; j < n; j++) {
                hv += h[(size_t)j*n+i]*c[q+1][j+1];
                sv += s[(size_t)j*n+i]*c[q+1][j+1];
            }
            residual += (hv-ko[q+1]*sv)*(hv-ko[q+1]*sv);
            norm += c[q+1][i+1]*c[q+1][i+1];
        }
        error = sqrt(residual) / ((scale_h + fabs(ko[q+1])*scale_s)*sqrt(norm));
        CHECK(isfinite(error) && error < 1e-12);
        if (worst_error < error) worst_error = error;
        for (int p = 0; p < k; p++) {
            double dot = 0;
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
                dot += c[p+1][i+1]*s[(size_t)j*n+i]*c[q+1][j+1];
            error = fabs(dot - (p == q));
            CHECK(isfinite(error) && error < 1e-12);
            if (worst_error < error) worst_error = error;
        }
    }
    numerical_checks++;
}

static void expect_fatal(int n, int k, double *s, double *h, double *ko, int kind)
{
    fatal_expected = 1; fatal_seen = 0;
    if (setjmp(fatal_jump) == 0) {
        (void)DCLNO_Solve_Col_GpuSolver(n, k, s, h, ko);
        CHECK(0 && "an invalid MAGMA result was accepted");
    }
    fatal_expected = 0;
    CHECK(fatal_seen == 1);
    /* a MAGMA error (a failed magma_init among them) is not reported as a
       rejected argument */
    if (kind == 2 || kind == 7) CHECK(strstr(fatal_message, "rejected an argument") != NULL);
    if (kind == 6) CHECK(strstr(fatal_message, "MAGMA error -116") != NULL &&
                         strstr(fatal_message, "rejected") == NULL);
    if (kind == 3) CHECK(strstr(fatal_message, "eigenpairs") != NULL);
}

static int retried(int kind) { return kind == 1 || kind == 4 || kind == 5; }

static void run_case(int n, int k, int stage, int kind)
{
    double s[MAX_N*MAX_N], h[MAX_N*MAX_N], s0[MAX_N*MAX_N], h0[MAX_N*MAX_N];
    double ko[MAX_N+2], c_storage[MAX_N+1][MAX_N+2], *c[MAX_N+1];
    double tmp[MAX_N*MAX_N], eig[MAX_N*MAX_N], work[1+6*MAX_N+2*MAX_N*MAX_N];
    int iwork[3+5*MAX_N];
    matrices(n, s, h);
    memcpy(s0, s, (size_t)n*n*sizeof(double));
    memcpy(h0, h, (size_t)n*n*sizeof(double));
    for (int i = 0; i < MAX_N+2; i++) ko[i] = sentinel;
    for (int j = 0; j <= MAX_N; j++) {
        c[j] = c_storage[j];
        for (int i = 0; i < MAX_N+2; i++) c[j][i] = sentinel;
    }
    fault_stage = stage; fault_kind = kind; solver_calls = 0;
    if (kind && !retried(kind)) {
        expect_fatal(n, k, s, h, ko, kind);
        CHECK(solver_calls == stage);
        CHECK(memcmp(s, s0, (size_t)n*n*sizeof(double)) == 0);
        CHECK(memcmp(h, h0, (size_t)n*n*sizeof(double)) == 0);
        failure_checks++;
    } else {
        int info = DCLNO_Solve_Col_GpuSolver(n, k, s, h, ko);
        CHECK(memcmp(s, s0, (size_t)n*n*sizeof(double)) == 0);
        if (kind) {
            CHECK(info == (kind == 1 ? 122 : DCLNO_RETRY_NO_MEMORY) && solver_calls == stage);
            CHECK(memcmp(h, h0, (size_t)n*n*sizeof(double)) == 0);
            /* the caller solves the complete generalized problem again on
               its CPU scratch from the untouched inputs */
            DCLNO_Solve_Col_Local(n, k, s, h, c, ko, tmp, eig, work, iwork);
            failure_checks++;
        } else {
            CHECK(info == 0 && solver_calls == 2);
            CHECK(memcmp(h + n*k, h0 + n*k, (size_t)n*(n-k)*sizeof(double)) == 0);
            DCLNO_CopyPackedEigvecsToC(h, n, k, c);
        }
        validate(n, k, s0, h0, ko, c);
        for (int j = 0; j <= n; j++) {
            CHECK(c[j][0] == sentinel && c[j][n+1] == sentinel);
            if (j == 0 || j > k) for (int i = 1; i <= n; i++) CHECK(c[j][i] == sentinel);
        }
    }
    /* the LAPACK helper uses ko[0:n] before shifting to one-based output */
    if (!retried(kind)) CHECK(ko[0] == sentinel);
    CHECK(ko[n+1] == sentinel);
}

int main(void)
{
    const int sizes[] = {3, 7};
    for (int a = 0; a < 2; a++) {
        const int n = sizes[a], counts[] = {1, n-1, n};
        for (int b = 0; b < 3; b++) {
            const int k = counts[b];
            run_case(n, k, 0, 0);
            for (int stage = 1; stage <= 2; stage++) {
                for (int kind = 1; kind <= 7; kind++) {
                    run_case(n, k, stage, kind);
                    /* a later problem must not inherit a failed solve's
                       poisoned device matrices */
                    run_case(n, k, 0, 0);
                }
            }
        }
    }
    DCLNO_GpuSolver_Destroy();
    CHECK(device_allocations == 0);
    printf("PASS: %d injected failures, %d numerical checks; "
           "overlap/H INFO, MAGMA allocation failures, eigenpair count, other MAGMA errors, "
           "original inputs, CPU retry, recovery and output bounds; max error %.3g\n",
           failure_checks, numerical_checks, worst_error);
    return 0;
}
'''


def make_harness(source):
    context_end = source.index("} DCLNO_GpuSolverCtx;") + len("} DCLNO_GpuSolverCtx;")
    context_start = source.rfind("typedef struct {", 0, context_end)
    macros = ["DCLNO_MAGMA_ERR_HOST_ALLOC", "DCLNO_MAGMA_ERR_DEVICE_ALLOC", "DCLNO_RETRY_NO_MEMORY",
              "DCLNO_RETRY_GPU_ERROR", "DCLNO_MAGMA_ERR", "DCLNO_MAGMA_ERR_ILLEGAL_VALUE"]
    names = ["DCLNO_CheckedArrayBytes", "DCLNO_CheckedMulCount", "DCLNO_MallocArray",
             "DCLNO_CopyPackedEigvecsToC", "DCLNO_Eigen_lapack_d_reuse"]
    parts = [PREFIX, *(define(source, name) for name in macros),
             *(function(source, name) for name in names),
             source[context_start:context_end],
             "static DCLNO_GpuSolverCtx DCLNO_gpusolver_ctx = {0};"]
    names = ["DCLNO_GpuSolver_Destroy", "DCLNO_GpuSolver_Init",
             "DCLNO_GpuSolver_EnsureMatrixCapacity", "DCLNO_GpuSolver_Eigen",
             "DCLNO_RetryTest", "DCLNO_ReportGpuRetry",
             "DCLNO_Solve_Col_GpuSolver", "DCLNO_Solve_Col_Local"]
    parts += [function(source, name) for name in names]
    return "\n\n".join(parts + [HARNESS])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    with tempfile.TemporaryDirectory(prefix="openmx-dclno-retry-") as directory:
        harness = Path(directory) / "retry.c"
        binary = Path(directory) / "retry"
        harness.write_text(make_harness(source))
        subprocess.run([*shlex.split(args.cc), *flags, "-std=c11", str(harness),
                        *shlex.split(args.libs), "-lm", "-o", str(binary)], check=True)
        env = os.environ.copy()
        env.pop("OPENMX_DCLNO_RETRY_TEST", None)
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", BLIS_NUM_THREADS="1")
        subprocess.run([str(binary)], env=env, check=True)


if __name__ == "__main__":
    main()
