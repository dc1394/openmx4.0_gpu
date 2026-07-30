/**********************************************************************
  dlaf_cosma_bridge.h

  C bridge between OpenMX and the distributed GPU solver stack used by
  scf.eigen.lib=gpusolver2: DLA-Future 0.10 (eigensolver, ScaLAPACK-like
  C API on the pika runtime) and COSMA (pdgemm/pzgemm).

  The bridge keeps the DLA-Future/pika lifecycle and the BLACS-grid
  registration in one translation unit so that the cluster solvers only
  deal with plain C calls that mirror the embedded-ELPA calls they
  replace.

  Runtime knobs (environment):
    OPENMX_GS2_PIKA_THREADS  worker threads of the pika runtime per rank
                             (default: cores of the node / local ranks)
    OPENMX_GS2_PIKA_BIND     pika thread binding ("none" by default so
                             that several ranks sharing a node do not pin
                             their workers onto the same cores; any pika
                             --pika:bind value, or "default" to leave the
                             pika default)
    DLAF_*                   native DLA-Future tuning, e.g.
                             DLAF_UMPIRE_DEVICE_MEMORY_POOL_INITIAL_BLOCK_BYTES
                             (the bridge presets the umpire pools to
                             256 MiB blocks instead of the 1 GiB default,
                             so several ranks can share one GPU; export
                             the variable to override)
***********************************************************************/

#ifndef DLAF_COSMA_BRIDGE_H
#define DLAF_COSMA_BRIDGE_H

/* Distributed eigensolvers on ScaLAPACK block-cyclic layouts.
   a is destroyed, the eigenvectors land in z (DLA-Future always computes
   all n of them; nev is accepted for interface compatibility), ALL n
   eigenvalues in w (ascending) — the same contract as the embedded ELPA
   wrappers.  Return value: 0 on success, nonzero otherwise. */
int openmx_gs2_eigen_real(int n, int nev, double *a, int *desca, double *w, double *z,
                          int *descz);
int openmx_gs2_eigen_complex(int n, int nev, void *a, int *desca, double *w, void *z,
                             int *descz);

/* Release the DLA-Future grid bound to a BLACS context; call right
   before Cblacs_gridexit. */
void openmx_gs2_grid_free(int ictxt);

/* Shut the DLA-Future/pika runtime down (idempotent; called once at
   program end, before MPI_Finalize). */
void openmx_gs2_finalize(void);

/* COSMA drop-in replacements for the ScaLAPACK p?gemm calls; the
   argument list is identical to pdgemm_/pzgemm_ (complex matrices are
   passed as double* just like the dcomplex arrays behind pzgemm_). */
void cosma_pdgemm_(const char *trans_a, const char *trans_b, const int *m, const int *n,
                   const int *k, const double *alpha, const double *a, const int *ia,
                   const int *ja, const int *desca, const double *b, const int *ib,
                   const int *jb, const int *descb, const double *beta, double *c, const int *ic,
                   const int *jc, const int *descc);
void cosma_pzgemm_(const char *trans_a, const char *trans_b, const int *m, const int *n,
                   const int *k, const double *alpha, const double *a, const int *ia,
                   const int *ja, const int *desca, const double *b, const int *ib,
                   const int *jb, const int *descb, const double *beta, double *c, const int *ic,
                   const int *jc, const int *descc);

#endif /* DLAF_COSMA_BRIDGE_H */
