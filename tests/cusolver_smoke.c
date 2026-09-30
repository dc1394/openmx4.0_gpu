/* Standalone accuracy/partial-output check for the cuSOLVER compatibility layer.
   Build with the selected CUDA include/lib directories and
   source/openmx_cusolver_compat.c; link -lcusolver -lcudart -lm.
   Run both normally and with OPENMX_CUSOLVER_ALGORITHM=legacy,
   OPENMX_CUSOLVER_EMULATION=0 for comparison. */
#include "openmx_cusolver_compat.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)
#define SOLVER(call) do { cusolverStatus_t s = (call); if (s != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "%s: status=%d\n", #call, s); exit(1); } } while (0)

static void run_case(int n, int maxn, int complex_matrix)
{
    cusolverDnHandle_t h;
    cudaDataType type = complex_matrix ? CUDA_C_64F : CUDA_R_64F;
    int width = complex_matrix ? 2 : 1;
    int full_api = n == maxn && getenv("OPENMX_TEST_SYEVD") != NULL;
    const char *algorithm = getenv("OPENMX_CUSOLVER_ALGORITHM");
    /* Native SYEVDX requires n eigenvalue slots, including for a subset.
       The new two-stage adapter also accepts a tightly sized subset W. */
    const char *minimum = getenv("OPENMX_CUSOLVER_TWO_STAGE_MIN_N");
    int manual_stage = algorithm && (!strcmp(algorithm, "two-stage") ||
        !strcmp(algorithm, "2") || !strcmp(algorithm, "one-stage") || !strcmp(algorithm, "1"));
    if ((!algorithm || !*algorithm) && minimum && *minimum) manual_stage = 1;
    int wcount = manual_stage ? maxn : n;
    if (minimum && n < strtoll(minimum, NULL, 10)) wcount = n;
#if CUSOLVER_VERSION < 12304
    wcount = n;
#endif
    size_t elements = (size_t)n * n * width, db = 0, hb = 0;
    double *a = calloc(elements, sizeof(double));
    double *v = malloc(elements * sizeof(double));
    double *w = malloc((wcount + 1) * sizeof(double));
    double *da, *dw, lo = 0, hi = 0, norm2 = 0, residual2 = 0, orth = 0;
    void *work, *host;
    int *di, info;
    int64_t meig = 0;
    cudaEvent_t start, stop;
    float milliseconds;
    if (!a || !v || !w) exit(1);
    for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
        size_t p = ((size_t)j * n + i) * width;
        a[p] = 1.0 / (1.0 + abs(i - j)) + (i == j ? i * 0.01 : 0);
        norm2 += a[p] * a[p];
        if (complex_matrix) {
            a[p+1] = sin((i-j) * 0.1) / (1.0 + abs(i-j));
            norm2 += a[p+1] * a[p+1];
        }
    }
    CUDA(cudaMalloc((void **)&da, elements * sizeof(double)));
    CUDA(cudaMalloc((void **)&dw, (wcount + 1) * sizeof(double)));
    CUDA(cudaMalloc((void **)&di, sizeof(int)));
    SOLVER(cusolverDnCreate(&h));
    if (full_api) {
        SOLVER(cusolverDnXsyevd_bufferSize(h, NULL, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER, n, type, da, n, CUDA_R_64F, dw, type, &db, &hb));
    }
    else {
        SOLVER(cusolverDnXsyevdx_bufferSize(h, NULL, CUSOLVER_EIG_MODE_VECTOR,
            CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n, type, da, n,
            &lo, &hi, 1, maxn, &meig, CUDA_R_64F, dw, type, &db, &hb));
    }
    CUDA(cudaMalloc(&work, db ? db : 1));
    host = hb ? malloc(hb) : NULL;
    if (hb && !host) exit(1);
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));
    for (int repeat = 0; repeat < 2; ++repeat) {
        w[wcount] = 123456.0;
        CUDA(cudaMemcpy(da, a, elements * sizeof(double), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(dw + wcount, w + wcount, sizeof(double), cudaMemcpyHostToDevice));
        CUDA(cudaEventRecord(start, 0));
        if (full_api) {
            SOLVER(cusolverDnXsyevd(h, NULL, CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_LOWER, n, type, da, n, CUDA_R_64F, dw, type,
                work, db, host, hb, di));
            meig = n;
        }
        else {
            SOLVER(cusolverDnXsyevdx(h, NULL, CUSOLVER_EIG_MODE_VECTOR,
                CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n, type, da, n,
                &lo, &hi, 1, maxn, &meig, CUDA_R_64F, dw, type, work, db, host, hb, di));
        }
        CUDA(cudaEventRecord(stop, 0));
        CUDA(cudaEventSynchronize(stop));
        CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        CUDA(cudaMemcpy(&info, di, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(w, dw, (wcount + 1) * sizeof(double), cudaMemcpyDeviceToHost));
        if (info || meig != maxn || w[wcount] != 123456.0) {
            fprintf(stderr, "Invalid output: info=%d meig=%lld guard=%g\n", info, (long long)meig, w[wcount]);
            exit(1);
        }
    }
    CUDA(cudaMemcpy(v, da, elements * sizeof(double), cudaMemcpyDeviceToHost));
    int check = maxn < 16 ? maxn : 16;
    for (int j = 0; j < check; ++j) {
        double *ar = calloc(n, sizeof(double)), *ai = calloc(n, sizeof(double));
        if (!ar || !ai) exit(1);
        /* Traverse column-major A contiguously; strided access makes the
           residual check dominate large (8192+) GPU benchmark cases. */
        for (int k = 0; k < n; ++k) {
            size_t vp = ((size_t)j*n+k)*width;
            for (int i = 0; i < n; ++i) {
                size_t ap = ((size_t)k*n+i)*width;
                ar[i] += a[ap]*v[vp];
                if (complex_matrix) {
                    ar[i] -= a[ap+1]*v[vp+1];
                    ai[i] += a[ap]*v[vp+1]+a[ap+1]*v[vp];
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            size_t p = ((size_t)j*n+i)*width;
            double rr = ar[i]-w[j]*v[p];
            double ri = complex_matrix ? ai[i]-w[j]*v[p+1] : 0;
            residual2 += rr*rr + ri*ri;
        }
        free(ar); free(ai);
        for (int k = 0; k < check; ++k) {
            double rr = 0, ri = 0;
            for (int i = 0; i < n; ++i) {
                size_t p = ((size_t)j*n+i)*width, q = ((size_t)k*n+i)*width;
                rr += v[p]*v[q];
                if (complex_matrix) { rr += v[p+1]*v[q+1]; ri += v[p]*v[q+1]-v[p+1]*v[q]; }
            }
            rr -= j == k;
            double err = hypot(rr, ri);
            if (err > orth) orth = err;
        }
    }
    double residual = sqrt(residual2 / norm2);
    printf("%s n=%d maxn=%d residual=%.3e orth=%.3e solve_ms=%.3f\n",
        complex_matrix ? "complex" : "real", n, maxn, residual, orth, milliseconds);
    if (residual > 1e-11 || orth > 1e-11) exit(1);
    for (int i = 1; i < maxn; ++i) if (w[i] < w[i-1]) exit(1);
    SOLVER(cusolverDnDestroy(h));
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
    CUDA(cudaFree(da)); CUDA(cudaFree(dw)); CUDA(cudaFree(di)); CUDA(cudaFree(work));
    free(a); free(v); free(w); free(host);
}

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 257;
    int runtime, major, minor, patch;
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (n < 16) return 2;
    CUDA(cudaRuntimeGetVersion(&runtime));
    SOLVER(cusolverGetProperty(MAJOR_VERSION, &major));
    SOLVER(cusolverGetProperty(MINOR_VERSION, &minor));
    SOLVER(cusolverGetProperty(PATCH_LEVEL, &patch));
    printf("CUDA runtime=%d cuSOLVER=%d.%d.%d\n", runtime, major, minor, patch);
    for (int c = 0; c < 2; ++c) {
        run_case(n, n, c);
        run_case(n, 16, c);
    }
    return 0;
}
