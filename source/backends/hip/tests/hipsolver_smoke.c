/* Check of the generic-API shim source/hipsolver_compat.h, as a C translation
   unit sees it (the HIP port of the NVIDIA tests/cusolver_smoke.c):
   hipsolverDnXsyevd and hipsolverDnXsyevdx (RANGE_ALL, the first 16 of
   RANGE_I, and RANGE_V over an interval holding the whole spectrum, whose
   bounds are values of the W type) for HIP_R_32F, HIP_R_64F, HIP_C_32F and
   HIP_C_64F, with the
   residual and the orthogonality of the first 16 vectors, ascending
   eigenvalues, the eigenpair count h_meig and a sentinel after the n
   eigenvalue slots (the typed hipSOLVER API uses all n of them); then the
   error paths: a W type that does not match A, an unsupported type
   (HIPSOLVER_STATUS_NOT_SUPPORTED, 7 as in ROCm), a dimension or a workspace
   beyond INT_MAX (HIPSOLVER_STATUS_INVALID_VALUE, 3).
   The shim only dispatches: rocSOLVER's complex divide and conquer defect
   (NaN vectors for real-valued matrices) is handled by the wrappers and
   tested by tests/gpusolver_wrapper_smoke.c.
   Build and run: tests/run_gpusolver_smoke.sh */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hipsolver_compat.h"

#define CHECK_HIP(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e_)); exit(1); } } while (0)
#define CHECK_SOLVER(call) do { hipsolverStatus_t s_ = (call); if (s_ != HIPSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "%s: status=%d\n", #call, (int)s_); exit(1); } } while (0)

static const char *type_name(hipDataType t)
{
    switch (t) {
    case HIP_R_32F: return "R_32F";
    case HIP_R_64F: return "R_64F";
    case HIP_C_32F: return "C_32F";
    default:        return "C_64F";
    }
}

/* api 0: Xsyevd, 1: Xsyevdx RANGE_ALL, 2: Xsyevdx RANGE_I (first 16),
   3: Xsyevdx RANGE_V on (-100, 100], which holds every eigenvalue */
static void run_case(hipsolverDnHandle_t h, int n, hipDataType type, int api)
{
    const int    cplx = (type == HIP_C_32F || type == HIP_C_64F);
    const int    single = (type == HIP_R_32F || type == HIP_C_32F);
    const hipDataType wtype = single ? HIP_R_32F : HIP_R_64F;
    const size_t real_size = single ? sizeof(float) : sizeof(double);
    const int    width = cplx ? 2 : 1;
    const int    maxn = (api == 2) ? 16 : n;
    const int    check = maxn < 16 ? maxn : 16;
    const double tol = single ? 2e-4 : 1e-11;
    const size_t elements = (size_t)n * n * width;
    double      *a = calloc(elements, sizeof(double)), *v = malloc(elements * sizeof(double));
    double      *w = malloc((n + 1) * sizeof(double));
    void        *buf = malloc(elements * real_size), *wbuf = malloc((n + 1) * real_size);
    void        *da = NULL, *dw = NULL, *work = NULL;
    int         *dinfo = NULL, info = -1;
    size_t       dbytes = 0, hbytes = 0;
    int64_t      meig = -1;
    const hipsolverEigRange_t range = api == 1 ? HIPSOLVER_EIG_RANGE_ALL
                                    : (api == 2 ? HIPSOLVER_EIG_RANGE_I : HIPSOLVER_EIG_RANGE_V);
    double       vl = -100.0, vu = 100.0, norm2 = 0.0, residual2 = 0.0, orth = 0.0;
    float        vlf = -100.0f, vuf = 100.0f;
    const void  *pvl = single ? (const void *)&vlf : (const void *)&vl;
    const void  *pvu = single ? (const void *)&vuf : (const void *)&vu;

    if (!a || !v || !w || !buf || !wbuf) exit(1);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const size_t p = ((size_t)j * n + i) * width;

            a[p] = 1.0 / (1.0 + abs(i - j)) + (i == j ? i * 0.01 : 0);
            norm2 += a[p] * a[p];
            if (cplx) {
                a[p + 1] = sin((i - j) * 0.1) / (1.0 + abs(i - j));
                norm2 += a[p + 1] * a[p + 1];
            }
        }
    for (size_t p = 0; p < elements; ++p) {
        if (single) ((float *)buf)[p] = (float)a[p];
        else ((double *)buf)[p] = a[p];
    }
    CHECK_HIP(hipMalloc(&da, elements * real_size));
    CHECK_HIP(hipMalloc(&dw, (n + 1) * real_size));
    CHECK_HIP(hipMalloc((void **)&dinfo, sizeof(int)));
    CHECK_HIP(hipMemcpy(da, buf, elements * real_size, hipMemcpyHostToDevice));
    for (int k = 0; k <= n; ++k) {
        if (single) ((float *)wbuf)[k] = 123456.0f;
        else ((double *)wbuf)[k] = 123456.0;
    }
    CHECK_HIP(hipMemcpy(dw, wbuf, (n + 1) * real_size, hipMemcpyHostToDevice));

    if (api == 0) {
        CHECK_SOLVER(hipsolverDnXsyevd_bufferSize(h, NULL, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n,
                                                  type, da, n, wtype, dw, type, &dbytes, &hbytes));
    }
    else {
        CHECK_SOLVER(hipsolverDnXsyevdx_bufferSize(h, NULL, HIPSOLVER_EIG_MODE_VECTOR, range,
                                                   HIPBLAS_FILL_MODE_LOWER, n, type, da, n, pvl, pvu, 1, maxn, &meig,
                                                   wtype, dw, type, &dbytes, &hbytes));
        if (meig != maxn) {
            fprintf(stderr, "query h_meig=%lld expected %d\n", (long long)meig, maxn);
            exit(1);
        }
    }
    if (hbytes != 0 || dbytes == 0) {
        fprintf(stderr, "workspace query: device %zu host %zu bytes\n", dbytes, hbytes);
        exit(1);
    }
    CHECK_HIP(hipMalloc(&work, dbytes));
    meig = -1;
    if (api == 0) {
        CHECK_SOLVER(hipsolverDnXsyevd(h, NULL, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n, type, da, n,
                                       wtype, dw, type, work, dbytes, NULL, 0, dinfo));
        meig = n;
    }
    else {
        CHECK_SOLVER(hipsolverDnXsyevdx(h, NULL, HIPSOLVER_EIG_MODE_VECTOR, range,
                                        HIPBLAS_FILL_MODE_LOWER, n, type, da, n, pvl, pvu, 1, maxn, &meig, wtype, dw,
                                        type, work, dbytes, NULL, 0, dinfo));
    }
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(&info, dinfo, sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(buf, da, elements * real_size, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(wbuf, dw, (n + 1) * real_size, hipMemcpyDeviceToHost));
    for (size_t p = 0; p < elements; ++p) v[p] = single ? (double)((float *)buf)[p] : ((double *)buf)[p];
    for (int k = 0; k <= n; ++k) w[k] = single ? (double)((float *)wbuf)[k] : ((double *)wbuf)[k];
    if (info != 0 || meig != maxn || w[n] != 123456.0) {
        fprintf(stderr, "%s api=%d: info=%d meig=%lld sentinel=%g\n", type_name(type), api, info, (long long)meig,
                w[n]);
        exit(1);
    }

    for (int j = 0; j < check; ++j) {
        double *ar = calloc(n, sizeof(double)), *ai = calloc(n, sizeof(double));

        if (!ar || !ai) exit(1);
        for (int k = 0; k < n; ++k) {
            const size_t vp = ((size_t)j * n + k) * width;

            for (int i = 0; i < n; ++i) {
                const size_t ap = ((size_t)k * n + i) * width;

                ar[i] += a[ap] * v[vp];
                if (cplx) {
                    ar[i] -= a[ap + 1] * v[vp + 1];
                    ai[i] += a[ap] * v[vp + 1] + a[ap + 1] * v[vp];
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            const size_t p = ((size_t)j * n + i) * width;
            const double rr = ar[i] - w[j] * v[p], ri = cplx ? ai[i] - w[j] * v[p + 1] : 0.0;

            residual2 += rr * rr + ri * ri;
        }
        free(ar);
        free(ai);
        for (int k = 0; k < check; ++k) {
            double rr = 0.0, ri = 0.0;

            for (int i = 0; i < n; ++i) {
                const size_t p = ((size_t)j * n + i) * width, q = ((size_t)k * n + i) * width;

                rr += v[p] * v[q];
                if (cplx) {
                    rr += v[p + 1] * v[q + 1];
                    ri += v[p] * v[q + 1] - v[p + 1] * v[q];
                }
            }
            rr -= (j == k);
            if (hypot(rr, ri) > orth) orth = hypot(rr, ri);
        }
    }
    {
        const double residual = sqrt(residual2 / norm2);

        printf("%s %s n=%d maxn=%d residual=%.3e orth=%.3e\n", type_name(type),
               api == 0 ? "Xsyevd" : (api == 1 ? "Xsyevdx(ALL)" : (api == 2 ? "Xsyevdx(I)" : "Xsyevdx(V)")), n,
               maxn, residual, orth);
        if (!(residual <= tol) || !(orth <= tol)) exit(1);
    }
    for (int i = 1; i < maxn; ++i)
        if (w[i] < w[i - 1]) exit(1);

    CHECK_HIP(hipFree(da));
    CHECK_HIP(hipFree(dw));
    CHECK_HIP(hipFree(dinfo));
    CHECK_HIP(hipFree(work));
    free(a);
    free(v);
    free(w);
    free(buf);
    free(wbuf);
}

static void error_paths(hipsolverDnHandle_t h)
{
    size_t  dbytes = 1, hbytes = 1;
    int64_t meig = 1;
    double  vl = 0.0, vu = 0.0;
    int     ok = 1;

    /* complex A with a complex W, an unsupported type (HIP_R_16F = 2) */
    ok &= hipsolverDnXsyevd_bufferSize(h, NULL, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, 8, HIP_C_64F,
                                       NULL, 8, HIP_C_64F, NULL, HIP_C_64F, &dbytes, &hbytes) ==
          HIPSOLVER_STATUS_NOT_SUPPORTED;
    ok &= dbytes == 0 && hbytes == 0;
    ok &= hipsolverDnXsyevdx_bufferSize(h, NULL, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_EIG_RANGE_I,
                                        HIPBLAS_FILL_MODE_LOWER, 8, (hipDataType)2, NULL, 8, &vl, &vu, 1, 4, &meig,
                                        HIP_R_32F, NULL, (hipDataType)2, &dbytes, &hbytes) ==
          HIPSOLVER_STATUS_NOT_SUPPORTED;
    ok &= meig == 0;
    ok &= hipsolverDnXsyevdx(h, NULL, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_EIG_RANGE_I, HIPBLAS_FILL_MODE_LOWER, 8,
                             HIP_R_64F, NULL, 8, &vl, &vu, 1, 4, &meig, HIP_R_32F, NULL, HIP_R_64F, NULL, 0, NULL, 0,
                             NULL) == HIPSOLVER_STATUS_NOT_SUPPORTED;
    /* sizes beyond the int API */
    ok &= hipsolverDnXsyevd_bufferSize(h, NULL, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER,
                                       (int64_t)INT_MAX + 1, HIP_R_64F, NULL, (int64_t)INT_MAX + 1, HIP_R_64F, NULL,
                                       HIP_R_64F, &dbytes, &hbytes) == HIPSOLVER_STATUS_INVALID_VALUE;
    ok &= hipsolverDnXsyevd(h, NULL, HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, 8, HIP_C_64F, NULL, 8,
                            HIP_R_64F, NULL, HIP_C_64F, NULL, ((size_t)INT_MAX + 1) * 16, NULL, 0, NULL) ==
          HIPSOLVER_STATUS_INVALID_VALUE;
    ok &= hipsolverDnXsyevdx(h, NULL, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_EIG_RANGE_I, HIPBLAS_FILL_MODE_LOWER, 8,
                             HIP_R_64F, NULL, 8, &vl, &vu, 1, (int64_t)INT_MAX + 1, &meig, HIP_R_64F, NULL,
                             HIP_R_64F, NULL, 0, NULL, 0, NULL) == HIPSOLVER_STATUS_INVALID_VALUE;
    /* a range the library rejects comes back as its status */
    ok &= hipsolverDnXsyevdx_bufferSize(h, NULL, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_EIG_RANGE_I,
                                        HIPBLAS_FILL_MODE_LOWER, 8, HIP_R_64F, NULL, 8, &vl, &vu, 1, 9, &meig,
                                        HIP_R_64F, NULL, HIP_R_64F, &dbytes, &hbytes) ==
          HIPSOLVER_STATUS_INVALID_VALUE;
    printf("error paths: %s\n", ok ? "ok" : "FAILED");
    if (!ok) exit(1);
}

int main(int argc, char **argv)
{
    const int         n = argc > 1 ? atoi(argv[1]) : 257;
    const hipDataType types[4] = {HIP_R_32F, HIP_R_64F, HIP_C_32F, HIP_C_64F};
    hipsolverDnHandle_t h;

    setvbuf(stdout, NULL, _IOLBF, 0);
    if (n < 16) return 2;
    CHECK_SOLVER(hipsolverDnCreate(&h));
    for (int t = 0; t < 4; ++t)
        for (int api = 0; api < 4; ++api) run_case(h, n, types[t], api);
    error_paths(h);
    CHECK_SOLVER(hipsolverDnDestroy(h));
    printf("hipsolver shim smoke: all cases passed\n");
    return 0;
}
