/**********************************************************************
  OpenMX CPU eigensolvers backed by the current bundled ELPA C API.

  OpenMX historically called a private ELPA 2018 implementation and an
  older one-stage Fortran module.  These compatibility entry points keep
  the distributed matrix convention of those calls while using the same
  ELPA library as the GPU backend, with the CPU kernel chosen at run time.
***********************************************************************/

#include "openmx_elpa_cpu.h"

#include <elpa/elpa.h>
#include <mpi.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "openmx_env.h"
#include "openmx_stdout_level.h"

/* The 2-stage kernel.  ELPA_DEFAULT_{real,complex}_kernel in the environment
   is ELPA's own override and is left alone.  Otherwise the fastest kernel
   compiled into the bundled library that this CPU can execute, in ELPA's
   preference order (AVX-512, AVX2, AVX, SSE, generic): the library checks
   the CPU only with --enable-heterogenous-cluster-support, so a kernel the
   CPU lacks would trap inside the solver.  Every rank of a node chooses the
   same kernel; the generic kernels are always built.  No static state and
   no direct getenv: the global-state audit admits neither. */
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
/* CPUID and XCR0 directly: nvc compiles __builtin_cpu_supports but has no
   runtime symbol for it, and the OS must have enabled the YMM/ZMM state. */
static unsigned cpu_xcr0_low(void)
{
    unsigned eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    (void)edx;
    return eax;
}
static int cpu_isa_available(int level)   /* 4 AVX-512, 3 AVX2+FMA, 2 AVX, 1 SSE4.2, 0 any */
{
    unsigned a, b, c, d, b7 = 0, xcr0 = 0;
    int avx_state, avx512_state;

    if (level <= 0) return 1;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    if ((c >> 27) & 1) xcr0 = cpu_xcr0_low();          /* OSXSAVE */
    avx_state = (xcr0 & 0x6) == 0x6;                    /* XMM and YMM saved */
    avx512_state = (xcr0 & 0xe6) == 0xe6;               /* plus opmask and ZMM */
    {
        const unsigned c1 = c;                          /* leaf 1 ECX: FMA, AVX, SSE4.2 */
        if (!__get_cpuid_count(7, 0, &a, &b7, &c, &d)) b7 = 0;
        switch (level) {
        case 4: return avx512_state && ((b7 >> 16) & 1) && ((b7 >> 17) & 1) && ((b7 >> 30) & 1) && ((b7 >> 31) & 1);
        case 3: return avx_state && ((b7 >> 5) & 1) && ((c1 >> 12) & 1);
        case 2: return avx_state && ((c1 >> 28) & 1);
        default: return (c1 >> 20) & 1;
        }
    }
}
#else
static int cpu_isa_available(int level)
{
    return level <= 0;
}
#endif

static void cpu_select_kernel(elpa_t handle, int is_complex, MPI_Comm parent, int *error)
{
    const struct { int level; int real_kernel, complex_kernel; const char *name; } candidates[] = {
        { 4, ELPA_2STAGE_REAL_AVX512_BLOCK2, ELPA_2STAGE_COMPLEX_AVX512_BLOCK1, "AVX-512" },
        { 3, ELPA_2STAGE_REAL_AVX2_BLOCK2,   ELPA_2STAGE_COMPLEX_AVX2_BLOCK1,   "AVX2" },
        { 2, ELPA_2STAGE_REAL_AVX_BLOCK2,    ELPA_2STAGE_COMPLEX_AVX_BLOCK1,    "AVX" },
        { 1, ELPA_2STAGE_REAL_SSE_BLOCK2,    ELPA_2STAGE_COMPLEX_SSE_BLOCK1,    "SSE" },
        { 0, ELPA_2STAGE_REAL_GENERIC,       ELPA_2STAGE_COMPLEX_GENERIC,       "generic" },
    };
    const char *name = is_complex ? "complex_kernel" : "real_kernel";
    size_t i;

    if (*error != ELPA_OK) return;
    /* ELPA's own override (registry ids EXT_ELPA_DEFAULT_*_kernel, ELPA's mixed-case names): leave the
       library's default to it */
    if (openmx_env_at(is_complex ? OPENMX_ENV_EXT_ELPA_DEFAULT_complex_kernel
                                 : OPENMX_ENV_EXT_ELPA_DEFAULT_real_kernel) != NULL) return;
    for (i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        int status = ELPA_OK, rank = 0;
        if (!cpu_isa_available(candidates[i].level)) continue;
        elpa_set_integer(handle, name, is_complex ? candidates[i].complex_kernel : candidates[i].real_kernel, &status);
        if (status != ELPA_OK) continue;   /* not compiled into this library */
        /* per-solve diagnostics (level 3), like the other per-step lines */
        MPI_Comm_rank(parent, &rank);
        if (rank == 0 && OPENMX_STDOUT_LEVEL(3)) {
            printf("<ELPA> 2-stage %s kernel: %s\n", is_complex ? "complex" : "real", candidates[i].name);
            fflush(stdout);
        }
        return;
    }
    *error = ELPA_ERROR;
}

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

    /* ELPA defaults to one thread even in an OpenMP-enabled library.
       Do not create a nested team or violate the caller's MPI thread level. */
    int requested_threads = 1, mpi_threads = MPI_THREAD_SINGLE;
#ifdef _OPENMP
    if (!omp_in_parallel()) requested_threads = omp_get_max_threads();
#endif
    MPI_Query_thread(&mpi_threads);
    int elpa_threads = mpi_threads >= MPI_THREAD_SERIALIZED ? requested_threads : 1;
    if (elpa_threads < 1) elpa_threads = 1;
    cpu_set_integer(handle, "omp_threads", elpa_threads, &error);

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
    int configured_threads = 0;
    elpa_get_integer(handle, "omp_threads", &configured_threads, &error);
    error = cpu_collective_error(error, parent);
    if (error != ELPA_OK) cpu_abort("OpenMP thread query", error, rows, cols);
    if (OPENMX_STDOUT_LEVEL(3) || openmx_env_flag_at(OPENMX_ENV_GPU_VERBOSE, 0)) {
        int rank = 0;
        MPI_Comm_rank(parent, &rank);
        if (rank == 0) {
            printf("<ELPA> CPU OpenMP threads: requested=%d configured=%d MPI_thread=%d\n",
                   requested_threads, configured_threads, mpi_threads);
            fflush(stdout);
        }
    }
    cpu_set_integer(handle, "solver", solver, &error);
    if (solver == ELPA_SOLVER_2STAGE) cpu_select_kernel(handle, is_complex, parent, &error);
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
