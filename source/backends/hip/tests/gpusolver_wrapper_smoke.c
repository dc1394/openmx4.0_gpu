/* GPU check of the cached generic eigensolver wrappers (source/gpusolver_Syevdx.c)
   and of the dense FP64 entry points of source/magma_openmx.cu: the HIP/OpenMP
   port of the NVIDIA tests/gpusolver_wrapper_smoke.c, extended by the AMD cases.

   Matrices: the tridiagonal Laplacian (real) and its diagonal-unitary transform
   (complex), both with the eigenvalues 2-2cos(k pi/(n+1)); a complex matrix with
   all elements real and an exactly block-diagonal one, for which rocSOLVER's
   complex divide and conquer (ROCm 7.2) returns NaN vectors with info 0 unless
   the phase similarity is applied; and a matrix with a NaN, which must fail
   without touching the caller's arrays.

   Wrapper cases: host entries; _openmp entries with host arrays (the DC
   non-collinear call), hipMalloc'd device arrays and OpenMP-mapped host arrays;
   real and complex; the complete spectrum (divide and conquer) and a part
   (subset driver); repeated calls (cached context), cache release, two OpenMP
   threads and argument errors.  Every case checks the eigenvalues, the
   residual, norm and orthogonality of the vectors, the sentinel W[maxn] and the
   columns beyond maxn, which the wrappers leave untouched.

   MAGMA entry points: openmx_magma_dsyevdx_gpu / zheevdx_gpu and
   openmx_hipsolver_zheevd_gpu on device matrices (hipMalloc'd, and for the
   latter also an OpenMP-mapped host array, the band solver's call).

   tests/run_gpusolver_smoke.sh builds this file with the two production files
   and runs it under the environment variables of the solver policy
   (OPENMX_GPUSOLVER_RETRY_TEST, OPENMX_MAGMA_OZAKI2_RETRY, ...). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "hip_runtime_compat.h"

typedef struct { double r, i; } dcomplex;

int  gpusolver_Syevdx(double *A, double *W, int m, int MaxN);
int  gpusolver_Syevdx_openmp(double *A, double *W, int m, int MaxN);
int  gpusolver_Syevdx_Complex(dcomplex *A, double *W, int m, int MaxN);
int  gpusolver_Syevdx_Complex_openmp(dcomplex *A, double *W, int m, int MaxN);
void openmx_gpusolver_cache_release(void);
int  openmx_magma_dsyevdx_gpu(int n, int maxn, double *d_A, double *w, int *mout);
int  openmx_magma_zheevdx_gpu(int n, int maxn, void *d_A, double *w, int *mout);
int  openmx_hipsolver_zheevd_gpu(int n, int maxn, void *d_A, double *w, int *mout);
int  openmx_gpu_eigen_backend(int n, int maxn, int cplx);
int  openmx_magma_release_d_workspace(void);
int  openmx_magma_release_z_workspace(void);

int MYID_MPI_COMM_WORLD = 0;

/* the ranks sharing the GPU (set_hip_default_device_from_local_rank.c in
   OpenMX), which divide the device memory the retry copy of magma_openmx.cu
   may take: OPENMX_TEST_RANKS_PER_DEVICE=1000000 sends the copies to the host */
int openmx_gpu_ranks_per_device_noncollective(void)
{
    const char *v = getenv("OPENMX_TEST_RANKS_PER_DEVICE");

    return (v != NULL && *v != '\0') ? atoi(v) : 1;
}

#define SENTINEL 123456.0

enum { KIND_LAPLACE, KIND_REAL_VALUED, KIND_BLOCKS, KIND_NAN };
enum { ENTRY_HOST, ENTRY_OPENMP_HOST, ENTRY_OPENMP_DEVICE, ENTRY_OPENMP_MAPPED, ENTRY_MAGMA, ENTRY_HIPSOLVER,
       ENTRY_HIPSOLVER_MAPPED };

static const char *entry_name[] = {"host", "openmp(host arrays)", "openmp(device arrays)", "openmp(mapped arrays)",
                                   "magma entry", "hipsolver zheevd entry", "hipsolver zheevd entry(mapped array)"};

static void fail(const char *what, int n, int maxn)
{
    fprintf(stderr, "FAIL: %s (n=%d maxn=%d)\n", what, n, maxn);
    exit(1);
}

static int compare_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* column-major n x n matrix, complex when cplx (interleaved re, im) */
static void make_matrix(double *a, int n, int cplx, int kind)
{
    const int w = cplx ? 2 : 1, half = n / 2;

    memset(a, 0, sizeof(double) * (size_t)w * n * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double t = (i == j) ? 2.0 : (abs(i - j) == 1 ? -1.0 : 0.0);
            size_t p = ((size_t)j * n + i) * w;

            if (kind == KIND_BLOCKS && (i < half) != (j < half)) t = 0.0;
            if (!cplx) {
                a[p] = t;
            }
            else if (kind == KIND_LAPLACE) {
                a[p] = t * cos((i - j) * 0.17);
                a[p + 1] = t * sin((i - j) * 0.17);
            }
            else {
                a[p] = t;
            }
        }
    }
    if (kind == KIND_NAN) {
        a[w * 1] = NAN;                 /* (1,0) */
        a[(size_t)w * n] = NAN;         /* (0,1) */
    }
}

/* the lowest maxn exact eigenvalues */
static void exact_eigenvalues(double *e, int n, int maxn, int kind)
{
    double *all = malloc(sizeof(double) * n);
    const int half = n / 2;

    if (all == NULL) fail("allocation", n, maxn);
    if (kind == KIND_BLOCKS) {
        for (int k = 0; k < half; ++k) all[k] = 2 - 2 * cos((k + 1) * acos(-1.0) / (half + 1));
        for (int k = 0; k < n - half; ++k) all[half + k] = 2 - 2 * cos((k + 1) * acos(-1.0) / (n - half + 1));
        qsort(all, n, sizeof(double), compare_double);
    }
    else {
        for (int k = 0; k < n; ++k) all[k] = 2 - 2 * cos((k + 1) * acos(-1.0) / (n + 1));
    }
    memcpy(e, all, sizeof(double) * maxn);
    free(all);
}

/* eigenvalue error, residual, norm and orthogonality of the first maxn pairs
   (v: n x maxn, column major) against the matrix a; returns the worst
   eigenvalue error */
static double check_pairs(const double *a, const double *v, const double *w, int n, int maxn, int cplx, int kind,
                          const char *label)
{
    const int    wd = cplx ? 2 : 1;
    double      *e = malloc(sizeof(double) * maxn);
    double       worst = 0.0, worst_res = 0.0, worst_orth = 0.0;
    const int    northo = maxn < 8 ? maxn : 8;

    if (e == NULL) fail("allocation", n, maxn);
    exact_eigenvalues(e, n, maxn, kind);
    for (int k = 0; k < maxn; ++k) {
        double err = fabs(w[k] - e[k]), norm = 0.0, res = 0.0;

        if (!isfinite(w[k])) fail("non-finite eigenvalue", n, maxn);
        if (err > worst) worst = err;
        if (k > 0 && w[k] < w[k - 1]) fail("eigenvalues not ascending", n, maxn);
        for (int i = 0; i < n; ++i) {
            double rr = 0.0, ri = 0.0;
            const size_t p = ((size_t)k * n + i) * wd;

            if (!isfinite(v[p]) || (cplx && !isfinite(v[p + 1]))) fail("non-finite eigenvector", n, maxn);
            norm += v[p] * v[p] + (cplx ? v[p + 1] * v[p + 1] : 0.0);
            for (int j = 0; j < n; ++j) {
                const size_t ap = ((size_t)j * n + i) * wd, vp = ((size_t)k * n + j) * wd;

                if (cplx) {
                    rr += a[ap] * v[vp] - a[ap + 1] * v[vp + 1];
                    ri += a[ap] * v[vp + 1] + a[ap + 1] * v[vp];
                }
                else {
                    rr += a[ap] * v[vp];
                }
            }
            rr -= w[k] * v[p];
            if (cplx) ri -= w[k] * v[p + 1];
            res += rr * rr + ri * ri;
        }
        if (fabs(norm - 1.0) > 1e-11) fail("eigenvector norm", n, maxn);
        if (sqrt(res) > worst_res) worst_res = sqrt(res);
    }
    for (int k = 0; k < northo; ++k) {
        for (int l = k + 1; l < northo; ++l) {
            double rr = 0.0, ri = 0.0;

            for (int i = 0; i < n; ++i) {
                const size_t p = ((size_t)k * n + i) * wd, q = ((size_t)l * n + i) * wd;

                rr += v[p] * v[q] + (cplx ? v[p + 1] * v[q + 1] : 0.0);
                if (cplx) ri += v[p] * v[q + 1] - v[p + 1] * v[q];
            }
            if (hypot(rr, ri) > worst_orth) worst_orth = hypot(rr, ri);
        }
    }
    free(e);
    if (worst > 1e-11 || worst_res > 1e-10 || worst_orth > 1e-11) {
        fprintf(stderr, "%s: eigenvalue error %.3e residual %.3e orthogonality %.3e\n", label, worst, worst_res,
                worst_orth);
        fail("accuracy", n, maxn);
    }
    return worst;
}

static const char *kind_name(int kind)
{
    switch (kind) {
    case KIND_LAPLACE:     return "laplacian";
    case KIND_REAL_VALUED: return "real-valued";
    case KIND_BLOCKS:      return "block-diagonal";
    default:               return "with-NaN";
    }
}

static int call_wrapper(int entry, int cplx, double *a, double *w, int n, int maxn)
{
    if (entry == ENTRY_HOST) {
        return cplx ? gpusolver_Syevdx_Complex((dcomplex *)a, w, n, maxn) : gpusolver_Syevdx(a, w, n, maxn);
    }
    return cplx ? gpusolver_Syevdx_Complex_openmp((dcomplex *)a, w, n, maxn) : gpusolver_Syevdx_openmp(a, w, n, maxn);
}

/* one wrapper case, twice (the second call reuses the cached context) */
static void run_case(int n, int maxn, int cplx, int kind, int entry)
{
    const int    wd = cplx ? 2 : 1;
    const size_t abytes = sizeof(double) * (size_t)wd * n * n;
    double      *orig = malloc(abytes), *a = malloc(abytes), *w = malloc(sizeof(double) * (n + 1));
    double       worst = 0.0;
    char         label[160];

    if (orig == NULL || a == NULL || w == NULL) fail("allocation", n, maxn);
    snprintf(label, sizeof(label), "wrapper %s %s %s", cplx ? "complex" : "real", kind_name(kind), entry_name[entry]);
    make_matrix(orig, n, cplx, kind);
    for (int repeat = 0; repeat < 2; ++repeat) {
        int info = 0;

        memcpy(a, orig, abytes);
        for (int k = 0; k <= n; ++k) w[k] = SENTINEL + k;
        if (entry == ENTRY_OPENMP_DEVICE) {
            void *d_a = NULL, *d_w = NULL;

            if (hipMalloc(&d_a, abytes) != hipSuccess || hipMalloc(&d_w, sizeof(double) * (n + 1)) != hipSuccess ||
                hipMemcpy(d_a, a, abytes, hipMemcpyHostToDevice) != hipSuccess ||
                hipMemcpy(d_w, w, sizeof(double) * (n + 1), hipMemcpyHostToDevice) != hipSuccess)
                fail("device staging", n, maxn);
            info = call_wrapper(entry, cplx, (double *)d_a, (double *)d_w, n, maxn);
            if (hipMemcpy(a, d_a, abytes, hipMemcpyDeviceToHost) != hipSuccess ||
                hipMemcpy(w, d_w, sizeof(double) * (n + 1), hipMemcpyDeviceToHost) != hipSuccess)
                fail("device staging", n, maxn);
            (void)hipFree(d_a);
            (void)hipFree(d_w);
        }
        else if (entry == ENTRY_OPENMP_MAPPED) {
            const size_t na = (size_t)wd * n * n;
            const int    nw = n + 1;

#pragma omp target data map(tofrom : a[0 : na], w[0 : nw])
            {
                /* the host arrays are stale inside the region: scribble on
                   them, the wrapper must use the device copies */
                for (size_t p = 0; p < na; ++p) a[p] = -7.0;
                info = call_wrapper(entry, cplx, a, w, n, maxn);
            }
        }
        else {
            info = call_wrapper(entry, cplx, a, w, n, maxn);
        }

        if (kind == KIND_NAN) {
            if (info == 0) fail("a matrix with a NaN was solved", n, maxn);
            for (int k = 0; k <= n; ++k)
                if (w[k] != SENTINEL + k) fail("W changed by a failed solve", n, maxn);
            for (size_t p = 0; p < (size_t)wd * n * n; ++p)
                if (memcmp(&a[p], &orig[p], sizeof(double)) != 0) fail("A changed by a failed solve", n, maxn);
            printf("%s n=%d maxn=%d: failed as expected (info=%d), A and W untouched\n", label, n, maxn, info);
            break;
        }
        if (info != 0) {
            fprintf(stderr, "%s: info=%d\n", label, info);
            fail("wrapper info", n, maxn);
        }
        for (int k = maxn; k <= n; ++k)
            if (w[k] != SENTINEL + k) fail("W beyond maxn changed", n, maxn);
        for (size_t p = (size_t)wd * n * maxn; p < (size_t)wd * n * n; ++p)
            if (a[p] != orig[p]) fail("A beyond column maxn changed", n, maxn);
        {
            double e = check_pairs(orig, a, w, n, maxn, cplx, kind, label);
            if (e > worst) worst = e;
        }
    }
    if (kind != KIND_NAN)
        printf("%s n=%d maxn=%d max_eigenvalue_error=%.3e\n", label, n, maxn, worst);
    free(orig);
    free(a);
    free(w);
}

/* the device-matrix entry points of magma_openmx.cu; ENTRY_HIPSOLVER_MAPPED
   hands openmx_hipsolver_zheevd_gpu the OpenMP mapping of a host array, as
   the band solver does, which MAGMA (its retry) must not copy to or from
   pageable host memory */
static void run_device_entry(int n, int maxn, int cplx, int kind, int entry)
{
    const int    wd = cplx ? 2 : 1;
    const size_t abytes = sizeof(double) * (size_t)wd * n * n;
    double      *orig = malloc(abytes), *v = malloc(abytes), *w = malloc(sizeof(double) * (n + 1));
    void        *d_a = NULL;
    int          info, mout = -1;
    char         label[160];

    if (orig == NULL || v == NULL || w == NULL) fail("allocation", n, maxn);
    snprintf(label, sizeof(label), "%s %s %s", entry_name[entry], cplx ? "complex" : "real", kind_name(kind));
    make_matrix(orig, n, cplx, kind);
    for (int k = 0; k <= n; ++k) w[k] = SENTINEL;
    if (entry == ENTRY_HIPSOLVER_MAPPED) {
        const size_t na = (size_t)wd * n * n;

        memcpy(v, orig, abytes);
#pragma omp target data map(tofrom : v[0 : na])
        {
            d_a = omp_get_mapped_ptr(v, omp_get_default_device());
            if (d_a == NULL || d_a == (void *)v) fail("no device copy of the mapped array", n, maxn);
            info = openmx_hipsolver_zheevd_gpu(n, maxn, d_a, w, &mout);
        }
    }
    else {
        if (hipMalloc(&d_a, abytes) != hipSuccess || hipMemcpy(d_a, orig, abytes, hipMemcpyHostToDevice) != hipSuccess)
            fail("device staging", n, maxn);
        if (entry == ENTRY_HIPSOLVER) info = openmx_hipsolver_zheevd_gpu(n, maxn, d_a, w, &mout);
        else if (cplx) info = openmx_magma_zheevdx_gpu(n, maxn, d_a, w, &mout);
        else info = openmx_magma_dsyevdx_gpu(n, maxn, (double *)d_a, w, &mout);
        if (hipMemcpy(v, d_a, abytes, hipMemcpyDeviceToHost) != hipSuccess) fail("device staging", n, maxn);
        (void)hipFree(d_a);
    }
    if (info != 0 || mout != maxn) {
        fprintf(stderr, "%s: info=%d mout=%d\n", label, info, mout);
        fail("entry point result", n, maxn);
    }
    if (w[n] != SENTINEL) fail("w[n] changed", n, maxn);
    printf("%s n=%d maxn=%d max_eigenvalue_error=%.3e\n", label, n, maxn,
           check_pairs(orig, v, w, n, maxn, cplx, kind, label));
    free(orig);
    free(v);
    free(w);
}

int main(void)
{
    const char *expect = getenv("OPENMX_TEST_EXPECT_BACKEND");

    setvbuf(stdout, NULL, _IOLBF, 0);
    if (gpusolver_Syevdx(NULL, NULL, 0, 0) != 0 || gpusolver_Syevdx(NULL, NULL, 4, 5) != -1 ||
        gpusolver_Syevdx_Complex(NULL, NULL, -1, 0) != -1 || gpusolver_Syevdx_openmp(NULL, NULL, 3, 2) != -1)
        fail("argument checks", 0, 0);
    printf("backend(n=300, real) = %d, backend(n=300, complex) = %d (1 rocSOLVER, 0 MAGMA)\n",
           openmx_gpu_eigen_backend(300, 300, 0), openmx_gpu_eigen_backend(300, 20, 1));
    if (expect != NULL && *expect != '\0' && openmx_gpu_eigen_backend(300, 20, 1) != atoi(expect))
        fail("backend policy", 300, 20);

    for (int entry = ENTRY_HOST; entry <= ENTRY_OPENMP_MAPPED; ++entry) {
        for (int c = 0; c < 2; ++c) {
            run_case(64, 64, c, KIND_LAPLACE, entry);
            run_case(129, 7, c, KIND_LAPLACE, entry);
            openmx_gpusolver_cache_release();
            run_case(257, 7, c, KIND_LAPLACE, entry);
            run_case(257, 257, c, KIND_LAPLACE, entry);
        }
    }
    /* rocSOLVER's complex divide and conquer defect */
    for (int entry = ENTRY_HOST; entry <= ENTRY_OPENMP_HOST; ++entry) {
        run_case(200, 200, 1, KIND_REAL_VALUED, entry);
        run_case(200, 200, 1, KIND_BLOCKS, entry);
        run_case(200, 16, 1, KIND_REAL_VALUED, entry);
        run_case(200, 16, 1, KIND_BLOCKS, entry);
    }
    /* a failed solve leaves the caller's arrays alone */
    run_case(100, 100, 0, KIND_NAN, ENTRY_HOST);
    run_case(100, 10, 1, KIND_NAN, ENTRY_OPENMP_HOST);
    openmx_gpusolver_cache_release();

#pragma omp parallel num_threads(2)
    {
        run_case(64, 7, omp_get_thread_num() % 2, KIND_LAPLACE, ENTRY_HOST);
    }
    openmx_gpusolver_cache_release();

    run_device_entry(300, 300, 0, KIND_LAPLACE, ENTRY_MAGMA);
    run_device_entry(300, 20, 0, KIND_LAPLACE, ENTRY_MAGMA);
    run_device_entry(300, 300, 1, KIND_LAPLACE, ENTRY_MAGMA);
    run_device_entry(300, 20, 1, KIND_BLOCKS, ENTRY_MAGMA);
    run_device_entry(300, 300, 1, KIND_REAL_VALUED, ENTRY_HIPSOLVER);
    run_device_entry(300, 20, 1, KIND_BLOCKS, ENTRY_HIPSOLVER);
    run_device_entry(300, 300, 1, KIND_LAPLACE, ENTRY_HIPSOLVER_MAPPED);
    run_device_entry(300, 20, 1, KIND_BLOCKS, ENTRY_HIPSOLVER_MAPPED);
    if (openmx_magma_release_d_workspace() != 0 || openmx_magma_release_z_workspace() != 0)
        fail("workspace release", 0, 0);
    run_device_entry(300, 20, 0, KIND_LAPLACE, ENTRY_MAGMA);
    (void)openmx_magma_release_d_workspace();
    printf("gpusolver wrapper smoke: all cases passed\n");
    return 0;
}
