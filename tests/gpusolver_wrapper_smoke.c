/* Validate cached host/OpenACC wrappers, partial output and cache release.
   The Hermitian case is a diagonal-unitary transform of the tridiagonal
   Laplacian, so both cases have the analytic eigenvalues 2-2*cos(k*pi/(n+1)). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { double r, i; } dcomplex;
int32_t gpusolver_Syevdx(double *, double *, int32_t, int32_t);
int32_t gpusolver_Syevdx_Complex(dcomplex *, double *, int32_t, int32_t);
int32_t gpusolver_Syevdx_openacc(double *, double *, int32_t, int32_t);
int32_t gpusolver_Syevdx_Complex_openacc(dcomplex *, double *, int32_t, int32_t);
void openmx_gpusolver_cache_release(void);

static void run_case(int n, int maxn, int complex_matrix, int resident)
{
    double *a = calloc((size_t)n*n, sizeof(double));
    dcomplex *z = calloc((size_t)n*n, sizeof(dcomplex));
    double *w = calloc(maxn+1, sizeof(double));
    double worst = 0;
    if (!a || !z || !w) exit(1);
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
            double t = i == j ? 2.0 : abs(i-j) == 1 ? -1.0 : 0.0;
            a[j*n+i] = t;
            z[j*n+i].r = t*cos((i-j)*0.17);
            z[j*n+i].i = t*sin((i-j)*0.17);
        }
        w[maxn] = 123456.0;
        int info;
        if (resident && complex_matrix) {
#pragma acc data copy(z[0:n*n], w[0:maxn])
            { info = gpusolver_Syevdx_Complex_openacc(z, w, n, maxn); }
        }
        else if (resident) {
#pragma acc data copy(a[0:n*n], w[0:maxn])
            { info = gpusolver_Syevdx_openacc(a, w, n, maxn); }
        }
        else if (complex_matrix) info = gpusolver_Syevdx_Complex(z, w, n, maxn);
        else info = gpusolver_Syevdx(a, w, n, maxn);
        if (info || w[maxn] != 123456.0) {
            fprintf(stderr, "wrapper info=%d guard=%g\n", info, w[maxn]); exit(1);
        }
        for (int k = 0; k < maxn; ++k) {
            double exact = 2-2*cos((k+1)*acos(-1.0)/(n+1));
            double error = fabs(w[k]-exact);
            if (error > worst) worst = error;
            double norm = 0;
            for (int i = 0; i < n; ++i) {
                size_t p = (size_t)k*n+i;
                norm += complex_matrix ? z[p].r*z[p].r + z[p].i*z[p].i : a[p]*a[p];
            }
            if (fabs(norm-1) > 1e-11 || error > 1e-11) exit(1);
        }
    }
    printf("wrapper %s %s n=%d maxn=%d max_eigenvalue_error=%.3e\n",
           complex_matrix ? "complex" : "real", resident ? "OpenACC" : "host", n, maxn, worst);
    free(a); free(z); free(w);
}

int main(void)
{
    if (gpusolver_Syevdx(NULL, NULL, 0, 0) != 0 ||
        gpusolver_Syevdx(NULL, NULL, 4, 5) == 0) return 1;
    for (int resident = 0; resident < 2; ++resident)
        for (int c = 0; c < 2; ++c) {
            run_case(64, 64, c, resident);
            run_case(129, 7, c, resident);
            openmx_gpusolver_cache_release();
            run_case(257, 7, c, resident);
        }
    openmx_gpusolver_cache_release();
#ifdef _OPENMP
#pragma omp parallel num_threads(2)
    {
        run_case(64, 7, omp_get_thread_num() % 2, 0);
    }
#endif
    return 0;
}
