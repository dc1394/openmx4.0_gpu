/**********************************************************************
  elpa_cosma_bridge.h

  C bridge between OpenMX and the distributed GPU solver stack used by
  scf.eigen.lib=gpusolver2: ELPA 2026.02 with NVIDIA GPU kernels
  (eigensolver, modern object-based C API) and COSMA (pdgemm/pzgemm).

  The bridge keeps the ELPA handle cache and its lifecycle in one
  translation unit so that the cluster solvers only deal with plain C
  calls that mirror the embedded-ELPA calls they replace.

  GPU memory guard: ELPA aborts the whole run when a device allocation
  fails inside a solve, so before every solve the bridge estimates the
  per-rank device demand and falls back to ELPA's CPU kernels (same
  distributed algorithm, host BLAS) when the GPU cannot hold it.
  Environment knobs:
    OPENMX_GS2_ELPA_GPU        1 = always GPU (skip the preflight),
                               0 = always CPU; unset = decide from memory
    OPENMX_GS2_GPU_RESERVE_MB  device headroom kept free for the rest of
                               OpenMX (default 512)
    OPENMX_GS2_GPU_FACTOR      demand model: FACTOR x local matrix bytes
                               (default 3) ...
    OPENMX_GS2_GPU_FIXED_MB    ... plus this fixed head per rank (128)
    OPENMX_GS2_GPU_VERBOSE     1 = report the verdict of every handle
                               build even when the GPU is chosen
***********************************************************************/

#ifndef ELPA_COSMA_BRIDGE_H
#define ELPA_COSMA_BRIDGE_H

/* Distributed eigensolvers on ScaLAPACK block-cyclic layouts.
   a is destroyed, the nev lowest eigenvectors land in z, ALL n eigenvalues
   in w (ascending) — the same contract as the embedded ELPA wrappers.
   Return value: 0 on success, an ELPA error code otherwise. */
int openmx_gs2_eigen_real(int n, int nev, double *a, int *desca, double *w, double *z,
                          int *descz);
int openmx_gs2_eigen_complex(int n, int nev, void *a, int *desca, double *w, void *z,
                             int *descz);

/* Release the cached ELPA handles bound to a BLACS context; call right
   before Cblacs_gridexit. */
void openmx_gs2_grid_free(int ictxt);

/* Shut the ELPA library down (idempotent; called once at program end). */
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

#endif /* ELPA_COSMA_BRIDGE_H */
