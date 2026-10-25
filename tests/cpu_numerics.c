/* Exercise the selected MPI/BLACS, real/complex ScaLAPACK and the production
   ELPA1/ELPA2 adapters against analytically known matrices. */
#include "openmx_elpa_cpu.h"
#include <complex.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

extern void Cblacs_get(int, int, int *);
extern void Cblacs_gridinit(int *, const char *, int, int);
extern void Cblacs_gridexit(int);
extern void F77_NAME(descinit, DESCINIT)(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
extern void F77_NAME(pdpotrf, PDPOTRF)(const char *, int *, double *, int *, int *, int *, int *);
extern void F77_NAME(pzpotrf, PZPOTRF)(const char *, int *, void *, int *, int *, int *, int *);
#ifdef OPENMX_TEST_CXX_BOUNDARY
extern int openmx_test_cpu_cxx(void);
#endif

int main(int argc, char **argv)
{
    int rank, size, context, info, failed = 0, all_failed;
    int n = 16, block = 2, zero = 0, one = 1, desc[9];
    MPI_Comm rows, cols;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#ifdef OPENMX_TEST_CXX_BOUNDARY
    if (openmx_test_cpu_cxx() != 6) failed = 1;
#endif
    if (size != 2) MPI_Abort(MPI_COMM_WORLD, 2);
    int local_rows = n / size;
    double a[128], q[128], ev[16];
    double complex z[128], v[128];
    Cblacs_get(-1, 0, &context);
    Cblacs_gridinit(&context, "R", size, 1);
    F77_NAME(descinit, DESCINIT)(desc, &n, &n, &block, &block, &zero, &zero, &context, &local_rows, &info);
    if (info) MPI_Abort(MPI_COMM_WORLD, 3);
    /* A diagonal matrix makes each local Cholesky element independently
       checkable, including the imaginary part and the descriptor's LP64 ABI. */
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < local_rows; ++i) {
            int row = (i / block * size + rank) * block + i % block;
            a[j * local_rows + i] = row == j ? (j + 1.) * (j + 1.) : 0.;
            z[j * local_rows + i] = a[j * local_rows + i];
        }
    }
    F77_NAME(pdpotrf, PDPOTRF)("L", &n, a, &one, &one, desc, &info);
    if (info) failed = 1;
    F77_NAME(pzpotrf, PZPOTRF)("L", &n, z, &one, &one, desc, &info);
    if (info) failed = 1;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < local_rows; ++i) {
            int row = (i / block * size + rank) * block + i % block;
            if (row >= j) {
                double expected = row == j ? j + 1. : 0.;
                if (!isfinite(a[j * local_rows + i]) ||
                    !isfinite(creal(z[j * local_rows + i])) ||
                    !isfinite(cimag(z[j * local_rows + i])) ||
                    fabs(a[j * local_rows + i] - expected) > 1e-12 ||
                    cabs(z[j * local_rows + i] - expected) > 1e-12) failed = 1;
            }
        }
    }
    Cblacs_gridexit(context);
#ifndef OPENMX_TEST_SCALAPACK_ONLY
    MPI_Comm_dup(MPI_COMM_WORLD, &rows);
    MPI_Comm_split(MPI_COMM_WORLD, rank, 0, &cols);
    int frows = (int)MPI_Comm_c2f(rows), fcols = (int)MPI_Comm_c2f(cols);
    int parent = (int)MPI_Comm_c2f(MPI_COMM_WORLD);
    for (int stage = 1; stage <= 2; ++stage) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < local_rows; ++i) {
                int row = (i / block * size + rank) * block + i % block;
                int index = j * local_rows + i;
                a[index] = row == j ? 2. : abs(row - j) == 1 ? 1. : 0.;
                z[index] = row == j ? 2. : row == j + 1 ? I : row + 1 == j ? -I : 0.;
            }
        }
        if (stage == 1) {
            F77_NAME(solve_evp_real, SOLVE_EVP_REAL)(&n, &n, a, &local_rows, ev, q, &local_rows, &block, &frows, &fcols);
        } else {
            F77_NAME(elpa_solve_evp_real_2stage_double_impl, ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(
                &n, &n, a, &local_rows, ev, q, &local_rows, &block, &n, &frows, &fcols, &parent);
        }
        for (int k = 0; k < n; ++k) {
            double expected = 2. - 2. * cos((k + 1.) * acos(-1.) / (n + 1.));
            if (!isfinite(ev[k]) || fabs(ev[k] - expected) > 2e-11) failed = 1;
        }
        if (stage == 1) {
            F77_NAME(solve_evp_complex, SOLVE_EVP_COMPLEX)(&n, &n, z, &local_rows, ev, v, &local_rows, &block, &frows, &fcols);
        } else {
            F77_NAME(elpa_solve_evp_complex_2stage_double_impl, ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)(
                &n, &n, z, &local_rows, ev, v, &local_rows, &block, &n, &frows, &fcols, &parent);
        }
        for (int k = 0; k < n; ++k) {
            double expected = 2. - 2. * cos((k + 1.) * acos(-1.) / (n + 1.));
            if (!isfinite(ev[k]) || fabs(ev[k] - expected) > 2e-11) failed = 1;
        }
    }
    MPI_Comm_free(&cols);
    MPI_Comm_free(&rows);
#endif
    MPI_Allreduce(&failed, &all_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (!rank) printf("MPI/BLACS and LP64 real/complex numerical libraries: %s\n", all_failed ? "FAIL" : "PASS");
    MPI_Finalize();
    return all_failed;
}
