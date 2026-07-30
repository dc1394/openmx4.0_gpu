/**********************************************************************
  slate_bridge.h

  C interface of slate_bridge.cc, which drives the SLATE library
  (Software for Linear Algebra Targeting Exascale) for
  scf.eigen.lib=gpusolver2: the distributed multi-GPU/multi-rank
  eigensolver and distributed matrix products of the mainline
  collinear and non-collinear cluster calculations.

  All routines take ScaLAPACK-style arguments (block-cyclic
  distributions described by desc arrays living on a BLACS context);
  the bridge recovers the MPI communicator and the process grid from
  the context, wraps the local arrays as SLATE matrices without
  copying, and runs the SLATE routine on the GPUs.
***********************************************************************/

#ifndef OPENMX_SLATE_BRIDGE_H
#define OPENMX_SLATE_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Standard eigenproblem of a real-symmetric / complex-Hermitian
   block-cyclic matrix a (the full matrix must be filled in; the bridge
   reads the lower triangle).  All n eigenvalues are stored ascending in
   w[0..n-1] and all n eigenvectors in z (the callers use only the
   lowest nev columns; nev is accepted for interface compatibility with
   the other eigensolvers).  a is destroyed.  Returns 0 on success. */

int openmx_gs2_eigen_real(int n, int nev, double *a, int *desca,
                          double *w, double *z, int *descz);

int openmx_gs2_eigen_complex(int n, int nev, void *a, int *desca,
                             double *w, void *z, int *descz);

/* Distributed matrix products C = alpha*op(A)*op(B) + beta*C with the
   argument list of pdgemm_/pzgemm_ (the complex scalars/arrays of
   openmx_gs2_pzgemm are passed as double pairs).  Only whole-matrix
   operations (ia=ja=ib=jb=ic=jc=1) are supported. */

void openmx_gs2_pdgemm(const char *transa, const char *transb,
                       const int *m, const int *n, const int *k,
                       const double *alpha,
                       const double *a, const int *ia, const int *ja, const int *desca,
                       const double *b, const int *ib, const int *jb, const int *descb,
                       const double *beta,
                       double *c, const int *ic, const int *jc, const int *descc);

void openmx_gs2_pzgemm(const char *transa, const char *transb,
                       const int *m, const int *n, const int *k,
                       const double *alpha,
                       const double *a, const int *ia, const int *ja, const int *desca,
                       const double *b, const int *ib, const int *jb, const int *descb,
                       const double *beta,
                       double *c, const int *ic, const int *jc, const int *descc);

#ifdef __cplusplus
}
#endif

#endif /* OPENMX_SLATE_BRIDGE_H */
