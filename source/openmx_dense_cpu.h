#ifndef OPENMX_DENSE_CPU_H
#define OPENMX_DENSE_CPU_H
#ifdef __cplusplus
extern "C" {
#endif
/* Host column-major lower triangle; leading nev eigenvectors replace A.
   A and W are unchanged on failure. Linear-workspace LAPACK avoids the
   n*n LP64 workspace limit of divide-and-conquer eigensolvers. */
int openmx_dense_cpu_eigen(int n, int nev, int is_complex, void *a, int lda, double *w);
/* C=op(A)*op(B), beta=0. N/T/C, with ordinary BLAS leading dimensions. */
int openmx_dense_cpu_gemm(char ta, char tb, int is_complex, int m, int n, int k,
    const void *a, int lda, const void *b, int ldb, void *c, int ldc);
#ifdef __cplusplus
}
#endif
#endif
