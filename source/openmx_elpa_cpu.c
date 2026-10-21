/**********************************************************************
  OpenMX CPU eigensolvers backed by the current bundled ELPA C API.

  OpenMX historically called a private ELPA 2018 implementation and an
  older one-stage Fortran module.  These compatibility entry points keep
  the distributed matrix convention of those calls while using the same
  ELPA library as the GPU backend, with CPU kernels explicitly selected.
***********************************************************************/

#include "openmx_elpa_cpu.h"

#include <elpa/elpa.h>
#include <mpi.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cpu_numroc(int n, int block, int process, int processes)
{
    const int blocks = n / block;
    int local = (blocks / processes) * block;
    if (process < blocks % processes) local += block;
    if (process == blocks % processes) local += n % block;
    return local;
}

/* The original one-stage ABI has no communicator covering the whole
   matrix.  Two reductions propagate an error to its Cartesian grid. */
static int cpu_grid_error(int error, MPI_Comm rows, MPI_Comm cols)
{
    int row_error, grid_error;
    MPI_Allreduce(&error, &row_error, 1, MPI_INT, MPI_MIN, rows);
    MPI_Allreduce(&row_error, &grid_error, 1, MPI_INT, MPI_MIN, cols);
    return grid_error;
}

static int cpu_collective_error(int error, MPI_Comm parent)
{
    int result;
    MPI_Allreduce(&error, &result, 1, MPI_INT, MPI_MIN, parent);
    return result;
}

static void cpu_abort(const char *operation, int error,
                      MPI_Comm rows, MPI_Comm cols)
{
    int row, col;
    MPI_Comm_rank(rows, &row);
    MPI_Comm_rank(cols, &col);
    if (row == 0 && col == 0) {
        fprintf(stderr, "OpenMX CPU ELPA: %s failed: %s (%d)\n",
                operation, elpa_strerr(error), error);
        fflush(stderr);
    }
    /* These historical entry points cannot return an error to their
       callers.  Stop all application ranks instead of using invalid
       eigenvectors or leaving another grid in a later collective. */
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    abort();
}

/* Form the parent of a matrix grid without involving ranks outside it.
   This matters for independent k-point and spin subcommunicators.  ELPA
   calls the communicator varying the process-row coordinate "rows". */
static MPI_Comm cpu_grid_parent(MPI_Comm rows, MPI_Comm cols,
                                int nrows, int ncols)
{
    int rank, error;
    int *row_ranks, *grid_ranks;
    MPI_Group world_group, grid_group;
    MPI_Comm parent;

    row_ranks = malloc((size_t)nrows * sizeof(*row_ranks));
    grid_ranks = malloc((size_t)nrows * (size_t)ncols * sizeof(*grid_ranks));
    error = row_ranks && grid_ranks ? ELPA_OK : ELPA_ERROR;
    error = cpu_grid_error(error, rows, cols);
    if (error != ELPA_OK) cpu_abort("matrix communicator allocation", error, rows, cols);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Allgather(&rank, 1, MPI_INT, row_ranks, 1, MPI_INT, rows);
    MPI_Allgather(row_ranks, nrows, MPI_INT, grid_ranks, nrows, MPI_INT, cols);
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, nrows * ncols, grid_ranks, &grid_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, grid_group, 0, &parent);
    MPI_Group_free(&grid_group);
    MPI_Group_free(&world_group);
    free(grid_ranks);
    free(row_ranks);
    return parent;
}

static void cpu_set_integer(elpa_t handle, const char *name, int value, int *error)
{
    if (*error == ELPA_OK) elpa_set_integer(handle, name, value, error);
}

static void cpu_eigenvectors(int solver, int is_complex,
                            int na, int nev, void *a, int lda, double *ev,
                            void *q, int ldq, int nblk, int supplied_ncols,
                            MPI_Comm rows, MPI_Comm cols, MPI_Comm parent)
{
    elpa_t handle = NULL;
    int row, col, nrows, ncols, local_rows, local_cols, ld;
    int error, destroy_error, parent_owned = 0, packed;
    size_t element_bytes, matrix_bytes;
    void *matrix_a = a, *matrix_q = q;

    MPI_Comm_rank(rows, &row);
    MPI_Comm_rank(cols, &col);
    MPI_Comm_size(rows, &nrows);
    MPI_Comm_size(cols, &ncols);
    error = na > 0 && nev > 0 && nev <= na && nblk > 0 &&
            nrows <= INT_MAX / ncols ? ELPA_OK : ELPA_ERROR_SETUP;
    error = cpu_grid_error(error, rows, cols);
    if (error != ELPA_OK) cpu_abort("matrix dimensions", error, rows, cols);

    local_rows = cpu_numroc(na, nblk, row, nrows);
    local_cols = cpu_numroc(na, nblk, col, ncols);
    error = lda >= local_rows && ldq >= local_rows && ev != NULL &&
            (supplied_ncols < 0 || supplied_ncols >= local_cols) &&
            (local_rows == 0 || local_cols == 0 || (a != NULL && q != NULL))
            ? ELPA_OK : ELPA_ERROR_SETUP;
    error = cpu_grid_error(error, rows, cols);
    if (error != ELPA_OK) cpu_abort("local matrix storage", error, rows, cols);

    if (parent != MPI_COMM_NULL) {
        int size;
        MPI_Comm_size(parent, &size);
        /* Old two-stage ELPA also accepted a parent larger than the
           matrix grid.  Modern ELPA requires exactly the grid's ranks. */
        if (size != nrows * ncols) parent = MPI_COMM_NULL;
    }
    if (parent == MPI_COMM_NULL) {
        parent = cpu_grid_parent(rows, cols, nrows, ncols);
        parent_owned = 1;
    }

    /* Current ELPA's one-stage solver cannot handle empty process
       columns.  If an axis has empty coordinates, every nonempty
       coordinate owns at most one block, so removing the empty ranks
       leaves its block-cyclic layout unchanged.  Solve on that smaller
       grid and return all eigenvalues to the original grid. */
    {
        const int blocks = na / nblk + (na % nblk != 0);
        if (nrows > blocks || ncols > blocks) {
            MPI_Comm active_parent, active_rows, active_cols;
            int rank, root, root_candidate;
            const int active = local_rows > 0 && local_cols > 0;
            MPI_Comm_rank(parent, &rank);
            root_candidate = row == 0 && col == 0 ? rank : INT_MAX;
            MPI_Allreduce(&root_candidate, &root, 1, MPI_INT, MPI_MIN, parent);
            MPI_Comm_split(parent, active ? 0 : MPI_UNDEFINED,
                           col * nrows + row, &active_parent);
            if (active) {
                MPI_Comm_split(active_parent, col, row, &active_rows);
                MPI_Comm_split(active_parent, row, col, &active_cols);
                cpu_eigenvectors(solver, is_complex, na, nev, a, lda, ev,
                                 q, ldq, nblk, supplied_ncols,
                                 active_rows, active_cols, active_parent);
                MPI_Comm_free(&active_cols);
                MPI_Comm_free(&active_rows);
                MPI_Comm_free(&active_parent);
            }
            MPI_Bcast(ev, na, MPI_DOUBLE, root, parent);
            if (parent_owned) MPI_Comm_free(&parent);
            return;
        }
    }

    /* ELPA exposes one leading dimension for both matrices; OpenMX's
       original ABI exposes two.  Repack only when they differ or a rank
       has empty local storage. */
    packed = lda != ldq || lda == 0 || local_cols == 0 || a == NULL || q == NULL;
    ld = packed ? (local_rows > 0 ? local_rows : 1) : lda;
    element_bytes = is_complex ? sizeof(double complex) : sizeof(double);
    error = (size_t)ld <= SIZE_MAX / (size_t)(local_cols > 0 ? local_cols : 1) /
                          element_bytes ? ELPA_OK : ELPA_ERROR;
    error = cpu_collective_error(error, parent);
    if (error != ELPA_OK) cpu_abort("matrix allocation size", error, rows, cols);
    matrix_bytes = (size_t)ld * (size_t)(local_cols > 0 ? local_cols : 1) * element_bytes;
    if (packed) {
        matrix_a = calloc(1, matrix_bytes);
        matrix_q = calloc(1, matrix_bytes);
        error = matrix_a && matrix_q ? ELPA_OK : ELPA_ERROR;
    }
    /* Packing can differ between ranks when their leading dimensions
       differ.  Every rank must join the allocation-error collective. */
    error = cpu_collective_error(error, parent);
    if (error != ELPA_OK) cpu_abort("matrix allocation", error, rows, cols);
    if (packed) {
        int j;
        if (local_rows > 0) {
            for (j = 0; j < local_cols; ++j)
                memcpy((char *)matrix_a + (size_t)j * ld * element_bytes,
                       (char *)a + (size_t)j * lda * element_bytes,
                       (size_t)local_rows * element_bytes);
        }
    }

    /* Initialization is idempotent.  Each call creates and destroys its
       handle, so a GPU bridge uninit cannot invalidate a cached CPU
       handle or initialization flag.  The bridge owns global uninit. */
    error = cpu_collective_error(elpa_init(ELPA_API_VERSION), parent);
    if (error != ELPA_OK) cpu_abort("initialization", error, rows, cols);
    handle = elpa_allocate(&error);
    if (error == ELPA_OK && handle == NULL) error = ELPA_ERROR;
    error = cpu_collective_error(error, parent);
    if (error != ELPA_OK) cpu_abort("handle allocation", error, rows, cols);

    cpu_set_integer(handle, "na", na, &error);
    cpu_set_integer(handle, "nev", nev, &error);
    cpu_set_integer(handle, "local_nrows", ld, &error);
    cpu_set_integer(handle, "local_ncols", local_cols, &error);
    cpu_set_integer(handle, "nblk", nblk, &error);
    cpu_set_integer(handle, "mpi_comm_parent", (int)MPI_Comm_c2f(parent), &error);
    cpu_set_integer(handle, "process_row", row, &error);
    cpu_set_integer(handle, "process_col", col, &error);
    /* All three switches exist in current ELPA, including builds where
       a particular GPU backend was not enabled.  Zero is always valid. */
    cpu_set_integer(handle, "nvidia-gpu", 0, &error);
    cpu_set_integer(handle, "amd-gpu", 0, &error);
    cpu_set_integer(handle, "intel-gpu", 0, &error);
    error = cpu_collective_error(error, parent);
    if (error != ELPA_OK) cpu_abort("handle configuration", error, rows, cols);

    error = cpu_collective_error(elpa_setup(handle), parent);
    if (error != ELPA_OK) cpu_abort("setup", error, rows, cols);
    cpu_set_integer(handle, "solver", solver, &error);
    if (solver == ELPA_SOLVER_2STAGE) {
        cpu_set_integer(handle, is_complex ? "complex_kernel" : "real_kernel",
                        is_complex ? ELPA_2STAGE_COMPLEX_GENERIC : ELPA_2STAGE_REAL_GENERIC,
                        &error);
    }
    error = cpu_collective_error(error, parent);
    if (error != ELPA_OK) cpu_abort("CPU solver selection", error, rows, cols);

    /* These matrices always live on the host. The explicit-name dispatcher
       probes CUDA/HIP pointer attributes even with GPU kernels disabled;
       CUDA's dispatcher exits on a GPU-less node. Use the public host-array
       entry points (also used by ELPA's generic C API) for CPU recovery. */
    if (is_complex)
        elpa_eigenvectors_a_h_a_dc(handle, matrix_a, ev, matrix_q, &error);
    else
        elpa_eigenvectors_a_h_a_d(handle, matrix_a, ev, matrix_q, &error);
    error = cpu_collective_error(error, parent);
    if (error != ELPA_OK) cpu_abort("eigenvectors", error, rows, cols);

    elpa_deallocate(handle, &destroy_error);
    error = cpu_collective_error(destroy_error, parent);
    if (error != ELPA_OK) cpu_abort("handle deallocation", error, rows, cols);
    if (packed) {
        int j;
        if (local_rows > 0) {
            for (j = 0; j < local_cols; ++j)
                memcpy((char *)q + (size_t)j * ldq * element_bytes,
                       (char *)matrix_q + (size_t)j * ld * element_bytes,
                       (size_t)local_rows * element_bytes);
        }
        free(matrix_q);
        free(matrix_a);
    }
    if (parent_owned) MPI_Comm_free(&parent);
}

void F77_NAME(solve_evp_real, SOLVE_EVP_REAL)(
    int *na, int *nev, double *a, int *lda, double *ev, double *q, int *ldq,
    int *nblk, int *mpi_comm_rows, int *mpi_comm_cols)
{
    cpu_eigenvectors(ELPA_SOLVER_1STAGE, 0, *na, *nev, a, *lda, ev, q, *ldq,
                     *nblk, -1, MPI_Comm_f2c((MPI_Fint)*mpi_comm_rows),
                     MPI_Comm_f2c((MPI_Fint)*mpi_comm_cols), MPI_COMM_NULL);
}

void F77_NAME(solve_evp_complex, SOLVE_EVP_COMPLEX)(
    int *na, int *nev, void *a, int *lda, double *ev, void *q, int *ldq,
    int *nblk, int *mpi_comm_rows, int *mpi_comm_cols)
{
    cpu_eigenvectors(ELPA_SOLVER_1STAGE, 1, *na, *nev, a, *lda, ev, q, *ldq,
                     *nblk, -1, MPI_Comm_f2c((MPI_Fint)*mpi_comm_rows),
                     MPI_Comm_f2c((MPI_Fint)*mpi_comm_cols), MPI_COMM_NULL);
}

void F77_NAME(elpa_solve_evp_real_2stage_double_impl,
              ELPA_SOLVE_EVP_REAL_2STAGE_DOUBLE_IMPL)(
    int *na, int *nev, double *a, int *lda, double *ev, double *q, int *ldq,
    int *nblk, int *local_ncols, int *mpi_comm_rows, int *mpi_comm_cols,
    int *mpi_comm_parent)
{
    cpu_eigenvectors(ELPA_SOLVER_2STAGE, 0, *na, *nev, a, *lda, ev, q, *ldq,
                     *nblk, *local_ncols, MPI_Comm_f2c((MPI_Fint)*mpi_comm_rows),
                     MPI_Comm_f2c((MPI_Fint)*mpi_comm_cols),
                     MPI_Comm_f2c((MPI_Fint)*mpi_comm_parent));
}

void F77_NAME(elpa_solve_evp_complex_2stage_double_impl,
              ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)(
    int *na, int *nev, void *a, int *lda, double *ev, void *q, int *ldq,
    int *nblk, int *local_ncols, int *mpi_comm_rows, int *mpi_comm_cols,
    int *mpi_comm_parent)
{
    cpu_eigenvectors(ELPA_SOLVER_2STAGE, 1, *na, *nev, a, *lda, ev, q, *ldq,
                     *nblk, *local_ncols, MPI_Comm_f2c((MPI_Fint)*mpi_comm_rows),
                     MPI_Comm_f2c((MPI_Fint)*mpi_comm_cols),
                     MPI_Comm_f2c((MPI_Fint)*mpi_comm_parent));
}
