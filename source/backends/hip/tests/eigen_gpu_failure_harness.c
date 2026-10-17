/* Host harness of the GPU eigensolver failure handling in source/Eigen_lapack.c
   and source/EigenBand_lapack.c (no GPU, no MPI library): the production files
   are linked against stubs of the gpusolver wrappers and of MPI_Abort, built
   by tests/run_eigen_gpu_failure.py.

   A stub either solves with LAPACK (the wrappers' contract: the lowest MaxN
   eigenvalues in W[0..MaxN-1], the vectors in the first MaxN columns of A) or
   fails with the given code after scribbling NaN over its matrix argument (a
   wrapper that overwrote its input before failing, as the old ones did), and
   leaves W alone.  Checked:
     EigenBand_lapack (N >= GPU_CPU_SWITCH_NUM, scf.eigen.lib gpusolver): a
       failure falls back to Eigen_HH on the original matrix, with the
       eigenpairs of the CPU path; a success matches the CPU path.
     Eigen_lapack (Eigen_gpusolver_x), Eigen_gpusolver_d,
     Eigen_gpusolver_x_openmp2, Eigen_gpusolver_x_complex_openmp,
     EigenBand_lapack_openmp: every failure (a positive info included, which
       the old code ignored) ends in MPI_Abort(mpi_comm_level1, 1) before
       anything is stored: the caller's matrix and eigenvalues are unchanged.
*/
#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_common.h"

void Eigen_lapack(double **a, double *ko, int n, int EVmax);
void EigenBand_lapack(dcomplex **A, double *W, int N0, int MaxN, int ev_flag);
void EigenBand_lapack_openmp(dcomplex *A, double *W, int N0, int MaxN);
void Eigen_gpusolver_d(double **a, double *ko, int n, int EVmax);
void Eigen_gpusolver_x_openmp2(double *a, double *ko, int n0, int EVmax);
void Eigen_gpusolver_x_complex_openmp(dcomplex **a, double *ko, int n0, int EVmax);

void dsyevd_(const char *, const char *, const int *, double *, const int *, double *, double *, const int *, int *,
             const int *, int *);
void zheevd_(const char *, const char *, const int *, dcomplex *, const int *, double *, dcomplex *, const int *,
             double *, const int *, int *, const int *, int *);

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(2); } } while (0)

static jmp_buf abort_jump;
static int     abort_expected, abort_seen, abort_code;
static int     stub_result;        /* 0: solve, else the code to return */
static int     stub_calls;

int MPI_Abort(MPI_Comm comm, int errorcode)
{
    (void)comm;
    abort_seen++;
    abort_code = errorcode;
    if (!abort_expected) {
        fprintf(stderr, "FAIL: unexpected MPI_Abort(%d)\n", errorcode);
        exit(2);
    }
    longjmp(abort_jump, 1);
    return 0;
}

int MPI_Finalize(void)
{
    return 0;
}

/* the wrappers' contract on a host matrix (column major, lda m) */
static int stub_real(double *A, double *W, int m, int MaxN)
{
    stub_calls++;
    if (stub_result != 0) {
        for (size_t p = 0; p < (size_t)m * m; ++p) A[p] = NAN;
        return stub_result;
    }
    {
        double *w = malloc(sizeof(double) * m), wq = 0.0;
        int     lwork = -1, liwork = -1, iq = 0, info = 0;
        double *work;
        int    *iwork;

        dsyevd_("V", "L", &m, A, &m, w, &wq, &lwork, &iq, &liwork, &info);
        lwork = (int)wq + 1;
        liwork = iq + 1;
        work = malloc(sizeof(double) * lwork);
        iwork = malloc(sizeof(int) * liwork);
        dsyevd_("V", "L", &m, A, &m, w, work, &lwork, iwork, &liwork, &info);
        CHECK(info == 0);
        memcpy(W, w, sizeof(double) * MaxN);
        free(w);
        free(work);
        free(iwork);
    }
    return 0;
}

static int stub_complex(dcomplex *A, double *W, int m, int MaxN)
{
    stub_calls++;
    if (stub_result != 0) {
        for (size_t p = 0; p < (size_t)m * m; ++p) A[p].r = A[p].i = NAN;
        return stub_result;
    }
    {
        double  *w = malloc(sizeof(double) * m), rq = 0.0;
        dcomplex wq;
        int      lwork = -1, lrwork = -1, liwork = -1, iq = 0, info = 0;
        dcomplex *work;
        double  *rwork;
        int     *iwork;

        zheevd_("V", "L", &m, A, &m, w, &wq, &lwork, &rq, &lrwork, &iq, &liwork, &info);
        lwork = (int)wq.r + 1;
        lrwork = (int)rq + 1;
        liwork = iq + 1;
        work = malloc(sizeof(dcomplex) * lwork);
        rwork = malloc(sizeof(double) * lrwork);
        iwork = malloc(sizeof(int) * liwork);
        zheevd_("V", "L", &m, A, &m, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
        CHECK(info == 0);
        memcpy(W, w, sizeof(double) * MaxN);
        free(w);
        free(work);
        free(rwork);
        free(iwork);
    }
    return 0;
}

int gpusolver_Syevdx(double *A, double *W, int m, int MaxN) { return stub_real(A, W, m, MaxN); }
int gpusolver_Syevdx_openmp(double *A, double *W, int m, int MaxN) { return stub_real(A, W, m, MaxN); }
int gpusolver_Syevdx_Complex(dcomplex *A, double *W, int m, int MaxN) { return stub_complex(A, W, m, MaxN); }
int gpusolver_Syevdx_Complex_openmp(dcomplex *A, double *W, int m, int MaxN) { return stub_complex(A, W, m, MaxN); }

static double **real_matrix(int n)
{
    double **a = malloc(sizeof(double *) * (n + 1));

    for (int i = 0; i <= n; ++i) a[i] = calloc(n + 1, sizeof(double));
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j) a[i][j] = 1.0 / (1.0 + abs(i - j)) + (i == j ? 0.01 * i : 0.0);
    return a;
}

static dcomplex **complex_matrix(int n)
{
    dcomplex **a = malloc(sizeof(dcomplex *) * (n + 1));

    for (int i = 0; i <= n; ++i) a[i] = calloc(n + 1, sizeof(dcomplex));
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j) {
            a[i][j].r = 1.0 / (1.0 + abs(i - j)) + (i == j ? 0.01 * i : 0.0);
            a[i][j].i = sin(0.1 * (i - j)) / (1.0 + abs(i - j));
        }
    return a;
}

/* residual of the eigenpairs (1-based columns k of v, values w[k]) of h */
static double complex_residual(dcomplex **h, dcomplex **v, const double *w, int n, int maxn)
{
    double worst = 0.0;

    for (int k = 1; k <= maxn; ++k) {
        double res = 0.0, norm = 0.0;

        for (int i = 1; i <= n; ++i) {
            double rr = -w[k] * v[i][k].r, ri = -w[k] * v[i][k].i;

            for (int j = 1; j <= n; ++j) {
                rr += h[i][j].r * v[j][k].r - h[i][j].i * v[j][k].i;
                ri += h[i][j].r * v[j][k].i + h[i][j].i * v[j][k].r;
            }
            res += rr * rr + ri * ri;
            norm += v[i][k].r * v[i][k].r + v[i][k].i * v[i][k].i;
        }
        CHECK(fabs(norm - 1.0) < 1e-9);
        if (sqrt(res) > worst) worst = sqrt(res);
    }
    return worst;
}

static void copy_complex(dcomplex **dst, dcomplex **src, int n)
{
    for (int i = 0; i <= n; ++i) memcpy(dst[i], src[i], sizeof(dcomplex) * (n + 1));
}

static int complex_equal(dcomplex **a, dcomplex **b, int n)
{
    for (int i = 0; i <= n; ++i)
        if (memcmp(a[i], b[i], sizeof(dcomplex) * (n + 1)) != 0) return 0;
    return 1;
}

static void test_eigenband(int n, int maxn)
{
    const int  codes[4] = {0, 3, -2, -1007};
    dcomplex **h = complex_matrix(n), **a = complex_matrix(n);
    double    *wref = calloc(n + 2, sizeof(double)), *w = calloc(n + 2, sizeof(double));

    /* the CPU path as the reference */
    scf_eigen_lib_flag = ELPA1;
    copy_complex(a, h, n);
    EigenBand_lapack(a, wref, n, maxn, 1);
    printf("EigenBand_lapack CPU n=%d maxn=%d residual=%.3e\n", n, maxn, complex_residual(h, a, wref, n, maxn));

    scf_eigen_lib_flag = GPUSOLVER;
    for (int c = 0; c < 4; ++c) {
        double diff = 0.0;

        stub_result = codes[c];
        stub_calls = 0;
        copy_complex(a, h, n);
        for (int k = 0; k <= n + 1; ++k) w[k] = -99.0;
        EigenBand_lapack(a, w, n, maxn, 1);
        CHECK(stub_calls == 1);
        for (int k = 1; k <= maxn; ++k)
            if (fabs(w[k] - wref[k]) > diff) diff = fabs(w[k] - wref[k]);
        CHECK(diff < 1e-9);
        printf("EigenBand_lapack GPU %s (code %d): max |dW| vs CPU %.3e, residual %.3e\n",
               codes[c] ? "failure -> Eigen_HH" : "success", codes[c], diff, complex_residual(h, a, w, n, maxn));
    }
    stub_result = 0;
}

/* the aborting entry points: the call must end in MPI_Abort with the
   caller's arrays unchanged */
#define EXPECT_ABORT(call, unchanged) do {                                        \
    abort_expected = 1; abort_seen = 0;                                           \
    if (setjmp(abort_jump) == 0) { call; CHECK(!"no MPI_Abort"); }                \
    abort_expected = 0;                                                           \
    CHECK(abort_seen == 1 && abort_code == 1);                                    \
    CHECK(unchanged);                                                             \
} while (0)

static void test_aborts(int n, int maxn)
{
    const int  codes[3] = {3, -2, -1007};
    double   **ra = real_matrix(n), **rb = real_matrix(n);
    dcomplex **ca = complex_matrix(n), **cb = complex_matrix(n);
    double    *ko = malloc(sizeof(double) * (n + 2)), *flat = malloc(sizeof(double) * n * n);
    double    *flat0 = malloc(sizeof(double) * n * n);
    dcomplex  *cflat = malloc(sizeof(dcomplex) * n * n), *cflat0 = malloc(sizeof(dcomplex) * n * n);

    scf_eigen_lib_flag = GPUSOLVER;
    for (int c = 0; c < 3; ++c) {
        int same;

        stub_result = codes[c];
        for (int k = 0; k <= n + 1; ++k) ko[k] = -99.0;

        /* Eigen_lapack -> Eigen_gpusolver_x */
        EXPECT_ABORT(Eigen_lapack(ra, ko, n, maxn), 1);
        same = 1;
        for (int i = 0; i <= n; ++i) same &= memcmp(ra[i], rb[i], sizeof(double) * (n + 1)) == 0;
        for (int k = 0; k <= n + 1; ++k) same &= ko[k] == -99.0;
        CHECK(same);

        EXPECT_ABORT(Eigen_gpusolver_d(ra, ko, n, maxn), 1);
        same = 1;
        for (int i = 0; i <= n; ++i) same &= memcmp(ra[i], rb[i], sizeof(double) * (n + 1)) == 0;
        for (int k = 0; k <= n + 1; ++k) same &= ko[k] == -99.0;
        CHECK(same);

        for (size_t p = 0; p < (size_t)n * n; ++p) flat[p] = flat0[p] = 0.5 + (double)(p % 7);
        EXPECT_ABORT(Eigen_gpusolver_x_openmp2(flat, ko, n, maxn), 1);
        same = 1;
        for (int k = 0; k <= n + 1; ++k) same &= ko[k] == -99.0;
        CHECK(same);

        EXPECT_ABORT(Eigen_gpusolver_x_complex_openmp(ca, ko, n, maxn), complex_equal(ca, cb, n));
        same = 1;
        for (int k = 0; k <= n + 1; ++k) same &= ko[k] == -99.0;
        CHECK(same);

        for (size_t p = 0; p < (size_t)n * n; ++p) cflat[p].r = cflat0[p].r = 0.25 * (double)(p % 5), cflat[p].i = cflat0[p].i = 0.0;
        EXPECT_ABORT(EigenBand_lapack_openmp(cflat, ko, n, maxn), 1);
        same = 1;
        for (int k = 0; k <= n + 1; ++k) same &= ko[k] == -99.0;
        CHECK(same);

        printf("code %d: Eigen_lapack, Eigen_gpusolver_d, _x_openmp2, _x_complex_openmp, EigenBand_lapack_openmp "
               "abort with the caller's arrays unchanged\n", codes[c]);
    }
    stub_result = 0;

    /* and the success path of Eigen_lapack against its CPU path */
    {
        double  **ref = real_matrix(n);
        double   *koref = calloc(n + 2, sizeof(double));
        double    diff = 0.0;

        scf_eigen_lib_flag = ELPA1;
        stub_calls = 0;
        Eigen_lapack(ref, koref, n, maxn);
        CHECK(stub_calls == 0);
        scf_eigen_lib_flag = GPUSOLVER;
        Eigen_lapack(ra, ko, n, maxn);
        CHECK(stub_calls == 1);
        for (int k = 1; k <= maxn; ++k)
            if (fabs(ko[k] - koref[k]) > diff) diff = fabs(ko[k] - koref[k]);
        CHECK(diff < 1e-9);
        printf("Eigen_lapack GPU success: max |dko| vs CPU %.3e (ko[1] %.15f)\n", diff, ko[1]);
    }
}

int main(void)
{
    const int n = GPU_CPU_SWITCH_NUM;   /* the smallest order of the GPU paths */

    setvbuf(stdout, NULL, _IOLBF, 0);
    test_aborts(n, 24);
    test_eigenband(n, 24);
    printf("eigen GPU failure harness: all checks passed\n");
    return 0;
}
