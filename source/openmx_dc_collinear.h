#ifndef OPENMX_DC_COLLINEAR_H
#define OPENMX_DC_COLLINEAR_H

#include "openmx_common.h"
#include "lapack_prototypes.h"

/* Contiguous, one-based row-major matrices. Restrict to the columns S of
   the orthogonalizer that survived the overlap cutoff. The Fortran BLAS
   sees transposed views, so no copy or in-place transposition is needed. */
static inline int DCCol_CPU_SolveHamiltonian(int n, int p_min, int ld,
                                            double **s, double **h, double *eigenvalues, double **vectors)
{
  int active=n-p_min+1;
  double one=1.0, zero=0.0;
  char normal='N', transpose='T';
  double *s_active=&s[1][p_min], *h_values=&h[1][1], *work=&vectors[1][1];

  /* H S, then S^T (H S). H is scratch after the first multiplication. */
  F77_NAME(dgemm,DGEMM)(&normal,&normal,&active,&n,&n,&one,s_active,&ld,h_values,&ld,&zero,work,&ld);
  F77_NAME(dgemm,DGEMM)(&normal,&transpose,&active,&active,&n,&one,work,&ld,s_active,&ld,&zero,h_values,&ld);
  Eigen_lapack_cpu(h,eigenvalues,active,active);

  /* Back-transform to the original PAO basis. */
  F77_NAME(dgemm,DGEMM)(&normal,&normal,&active,&n,&active,&one,h_values,&ld,s_active,&ld,&zero,work,&ld);
  return active;
}

#endif
