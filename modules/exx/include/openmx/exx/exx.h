#ifndef OPENMX_EXX_EXX_H
#define OPENMX_EXX_EXX_H

/* C interface of the exact exchange of OpenMX with the local RI (see the
   headers *.hpp for the physics and the conventions, and
   docs/EXX_IMPLEMENTATION_GUIDE.md for the equations).  Functions that
   return int return 0 on success. */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- matrix products (section 7.2 of the guide) ------------------------- */

/* dgemm computes C = alpha op(A) op(B) + beta C for column-major arrays with
   the arguments of DGEMM of the BLAS (transa, transb: 'N' or 'T'), and
   returns 0 on success.  Whether it runs on the host or on a device is its
   own business; the arrays it receives are host memory.  context is passed
   through unchanged. */
typedef struct {
  int (*dgemm)(void *context, char transa, char transb, int m, int n, int k, double alpha,
               const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc);
  void *context;
} OpenMXExxDenseBackend;

/* The product that the library uses when no table is given: plain loops.
   context is ignored.  Returns 1 for arguments that DGEMM would reject. */
int openmx_exx_reference_dgemm(void *context, char transa, char transb, int m, int n, int k, double alpha,
                               const double *a, int lda, const double *b, int ldb, double beta, double *c,
                               int ldc);

#ifdef __cplusplus
}
#endif

#endif
