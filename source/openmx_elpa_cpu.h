#ifndef OPENMX_ELPA_CPU_H
#define OPENMX_ELPA_CPU_H

#include "f77func.h"

/* Keep the calling convention used by OpenMX's CPU eigensolvers.  The
   implementation uses the same current ELPA library as gpusolver2.  Complex
   matrices contain interleaved real/imaginary doubles (OpenMX dcomplex). */
void F77_NAME(solve_evp_real, SOLVE_EVP_REAL)(
    int *na, int *nev, double *a, int *lda, double *ev, double *q, int *ldq,
    int *nblk, int *mpi_comm_rows, int *mpi_comm_cols);
void F77_NAME(solve_evp_complex, SOLVE_EVP_COMPLEX)(
    int *na, int *nev, void *a, int *lda, double *ev, void *q, int *ldq,
    int *nblk, int *mpi_comm_rows, int *mpi_comm_cols);
void F77_NAME(elpa_solve_evp_real_2stage_double_impl,
              ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(
    int *na, int *nev, double *a, int *lda, double *ev, double *q, int *ldq,
    int *nblk, int *local_ncols, int *mpi_comm_rows, int *mpi_comm_cols,
    int *mpi_comm_parent);
void F77_NAME(elpa_solve_evp_complex_2stage_double_impl,
              ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)(
    int *na, int *nev, void *a, int *lda, double *ev, void *q, int *ldq,
    int *nblk, int *local_ncols, int *mpi_comm_rows, int *mpi_comm_cols,
    int *mpi_comm_parent);

#endif
