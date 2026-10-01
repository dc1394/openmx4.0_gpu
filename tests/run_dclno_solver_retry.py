#!/usr/bin/env python3
"""Check DC-LNO solver failure handling without CUDA or MPI.

Production helpers are extracted verbatim. Host CUDA stubs inject solver INFO
and eigenpair-count failures; successful solves and the CPU retry use real
LAPACK/BLAS. This checks control flow, input preservation and numerical results,
not real cuSOLVER convergence or MPI/GPU execution. --sanitize adds ASan/UBSan.
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


PREFIX = r'''
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); \
} } while (0)
typedef int INTEGER;
typedef void *cudaStream_t;
typedef void *cublasHandle_t;
typedef void *cusolverDnHandle_t;
typedef int cusolverEigMode_t;
typedef int cublasFillMode_t;
typedef int cusolverEigRange_t;
typedef int cusolverStatus_t;
enum { CUSOLVER_STATUS_SUCCESS = 0 };
enum { cudaSuccess, cudaStreamNonBlocking, cudaMemcpyHostToDevice,
       cudaMemcpyDeviceToHost, CUSOLVER_EIG_MODE_VECTOR,
       CUBLAS_FILL_MODE_LOWER, CUSOLVER_EIG_RANGE_ALL,
       CUSOLVER_EIG_RANGE_I, CUDA_R_64F, CUBLAS_SIDE_RIGHT,
       CUBLAS_OP_N, CUBLAS_OP_C };
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
static int fault_stage, fault_kind, solver_calls, device_allocations;
static const double sentinel = -12345.25;
static void DCLNO_AbortWithMessage(const char *message)
{
    if (!fatal_expected) { fprintf(stderr, "Unexpected fatal: %s\n", message); exit(3); }
    fatal_seen++;
    longjmp(fatal_jump, 1);
}
#define wait_cudafunc(call) CHECK((call) == 0)
static int cudaStreamCreateWithFlags(cudaStream_t *p, int flags) { *p = (void *)1; return 0; }
static int cublasCreate(cublasHandle_t *p) { *p = (void *)2; return 0; }
static int cusolverDnCreate(cusolverDnHandle_t *p) { *p = (void *)3; return 0; }
static int cublasSetStream(cublasHandle_t h, cudaStream_t s) { return 0; }
static int cusolverDnSetStream(cusolverDnHandle_t h, cudaStream_t s) { return 0; }
static int cudaStreamSynchronize(cudaStream_t s) { return 0; }
static int cudaStreamDestroy(cudaStream_t s) { return 0; }
static int cublasDestroy(cublasHandle_t h) { return 0; }
static int cusolverDnDestroy(cusolverDnHandle_t h) { return 0; }
static int cudaMalloc(void **p, size_t bytes)
{
    *p = malloc(bytes ? bytes : 1); CHECK(*p != NULL); device_allocations++; return 0;
}
static int cudaFree(void *p) { if (p) { free(p); device_allocations--; } return 0; }
static int cudaMemcpyAsync(void *to, const void *from, size_t bytes, int kind, cudaStream_t stream)
{
    memcpy(to, from, bytes); return 0;
}
static int cusolverDnXsyevdx_bufferSize(cusolverDnHandle_t handle, void *params,
    int jobz, int range, int uplo, int64_t n, int type_a, void *a, int64_t lda,
    const void *vl, const void *vu, int64_t il, int64_t iu, int64_t *meig,
    int type_w, void *w, int compute, size_t *device_bytes, size_t *host_bytes)
{
    /* Nonzero buffers exercise production reuse and cleanup. */
    if (fault_kind == 5 && solver_calls + 1 == fault_stage) return 7;
    *device_bytes = 64; *host_bytes = 64; return 0;
}
static int cusolverDnXsyevdx(cusolverDnHandle_t handle, void *params,
    int jobz, int range, int uplo, int64_t n64, int type_a, void *a_void, int64_t lda64,
    const void *vl, const void *vu, int64_t il, int64_t iu, int64_t *meig,
    int type_w, void *w_void, int compute, void *device_work, size_t device_bytes,
    void *host_work, size_t host_bytes, int32_t *info)
{
    double *a = a_void, *w = w_void;
    int n = (int)n64, lda = (int)lda64;
    const int count = range == CUSOLVER_EIG_RANGE_ALL ? n : (int)(iu - il + 1);
    solver_calls++;
    *meig = count;
    if (solver_calls == fault_stage && fault_kind) {
        if (fault_kind == 4) return 7; /* API error must remain fatal. */
        /* Failed device solves are allowed to destroy their input/output. */
        for (int i = 0; i < n * n; i++) a[i] = NAN;
        for (int i = 0; i < n; i++) w[i] = NAN;
        *info = fault_kind == 1 ? 122 : fault_kind == 2 ? -7 : 0;
        if (fault_kind == 3) *meig = count - 1;
        return 0;
    }
    int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n, result;
    double *work = malloc((size_t)lwork * sizeof(*work));
    int *iwork = malloc((size_t)liwork * sizeof(*iwork));
    CHECK(work && iwork);
    dsyevd_("V", "L", &n, a, &lda, w, work, &lwork, iwork, &liwork, &result);
    free(iwork); free(work); *info = result; return 0;
}
static int cublasDdgmm(cublasHandle_t handle, int side, int m, int n,
    const double *a, int lda, const double *x, int incx, double *c, int ldc)
{
    CHECK(side == CUBLAS_SIDE_RIGHT);
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++)
        c[(size_t)j*ldc+i] = a[(size_t)j*lda+i] * x[j*incx];
    return 0;
}
static int openmx_gemmul8Dgemm(cublasHandle_t handle, int op_a, int op_b,
    int m, int n, int k, const double *alpha, const double *a, int lda,
    const double *b, int ldb, const double *beta, double *c, int ldc)
{
    const char *ta = op_a == CUBLAS_OP_N ? "N" : "T";
    const char *tb = op_b == CUBLAS_OP_N ? "N" : "T";
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

static void expect_fatal(int n, int k, double *s, double *h, double *ko)
{
    /* Keep caller-owned matrices outside the function containing setjmp;
       their contents remain defined after longjmp even if a buggy solver
       overwrites them before rejecting the injected error. */
    fatal_expected = 1; fatal_seen = 0;
    if (setjmp(fatal_jump) == 0) {
        (void)DCLNO_Solve_Col_GpuSolver(n, k, s, h, ko);
        CHECK(0 && "Invalid INFO/count/API status was accepted");
    }
    fatal_expected = 0;
    CHECK(fatal_seen == 1);
}

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
    if (kind >= 2) {
        expect_fatal(n, k, s, h, ko);
        CHECK(solver_calls == stage - (kind == 5));
        CHECK(memcmp(s, s0, (size_t)n*n*sizeof(double)) == 0);
        CHECK(memcmp(h, h0, (size_t)n*n*sizeof(double)) == 0);
        failure_checks++;
    } else {
        int info = DCLNO_Solve_Col_GpuSolver(n, k, s, h, ko);
        CHECK(memcmp(s, s0, (size_t)n*n*sizeof(double)) == 0);
        if (kind == 1) {
            CHECK(info == 122 && solver_calls == stage);
            CHECK(memcmp(h, h0, (size_t)n*n*sizeof(double)) == 0);
            /* The caller retries the complete generalized problem with its
               existing CPU workspace, using the untouched original inputs. */
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
    /* LAPACK's CPU helper uses ko[0:n] before shifting to one-based output. */
    if (kind != 1) CHECK(ko[0] == sentinel);
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
                for (int kind = 1; kind <= 5; kind++) {
                    run_case(n, k, stage, kind);
                    /* A later problem must not inherit a failed solve's
                       poisoned device matrix, INFO or cached workspace. */
                    run_case(n, k, 0, 0);
                }
            }
        }
    }
    DCLNO_GpuSolver_Destroy();
    CHECK(device_allocations == 0);
    printf("PASS: %d injected failures, %d numerical checks; "
           "overlap/H INFO, eigenpair count, API failure, original inputs, "
           "CPU retry, recovery and output bounds; max error %.3g\n",
           failure_checks, numerical_checks, worst_error);
    return 0;
}
'''


def make_harness(source):
    context_end = source.index("} DCLNO_GpuSolverCtx;") + len("} DCLNO_GpuSolverCtx;")
    context_start = source.rfind("typedef struct {", 0, context_end)
    names = ["DCLNO_CheckedArrayBytes", "DCLNO_CheckedMulCount", "DCLNO_MallocArray",
             "DCLNO_CopyPackedEigvecsToC", "DCLNO_Eigen_lapack_d_reuse"]
    parts = [PREFIX, *(function(source, name) for name in names),
             source[context_start:context_end],
             "static DCLNO_GpuSolverCtx DCLNO_gpusolver_ctx = {0};"]
    names = ["DCLNO_GpuSolver_Destroy", "DCLNO_GpuSolver_Init",
             "DCLNO_GpuSolver_EnsureMatrixCapacity", "DCLNO_GpuSolver_EnsureWorkspace",
             "DCLNO_GpuSolver_Eigen", "DCLNO_Solve_Col_GpuSolver", "DCLNO_Solve_Col_Local"]
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
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", BLIS_NUM_THREADS="1")
        subprocess.run([str(binary)], env=env, check=True)


if __name__ == "__main__":
    main()
