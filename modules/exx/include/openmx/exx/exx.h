#ifndef OPENMX_EXX_EXX_H
#define OPENMX_EXX_EXX_H

/* C interface of the exact exchange of OpenMX with the local RI (see the
   headers *.hpp for the physics and the conventions, and
   docs/EXX_IMPLEMENTATION_GUIDE.md for the equations).

   Units are Hartree atomic units: lengths in bohr, wave numbers in 1/bohr.
   Matrices of orbitals are N x N arrays, N the number of orbitals of the unit
   cell; the orbitals are ordered by atom and, within an atom, as in the main
   program.  Nothing here uses MPI or keeps global state: all data live in the
   OpenMXExxState of the caller.  Functions that return int return 0 on
   success.  On failure they return a nonzero value and, where they take
   "error" and "capacity", write a terminated message of at most capacity
   bytes; error may be NULL. */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- the device of the exchange build (section 11 of the guide) --------- */

/* One side of a move on the device: element (r, c) lies at the offset of row
   r, r*stride or rows[r], plus the offset of column c, c*step or columns[c].
   rows and columns are NULL or arrays in device memory.  Offsets count
   doubles. */
typedef struct {
  long long stride;
  const long long *rows;
  long long step;
  const long long *columns;
} OpenMXExxDeviceIndex;

/* A device on which openmx_exx_add_blocks and openmx_exx_add_gamma compute
   the contributions of all their centre atoms: the density matrix, the
   exchange matrix, the T and V blocks and the arrays of one centre atom stay
   in device memory, and the four terms of a V neighbour are one product each
   for all ABFs of the centre (section 11).  The host then makes no product
   of the call.  The functions are called from the thread that called the
   library, in this order of their use:

   accept is called first and once for a call with centre atoms: whether
     the device takes a call whose largest product has `multiplications`
     multiplications and whose arrays take `bytes` bytes of device memory,
     the sum of the allocations that follow.  1 takes it, 0 leaves the call
     to the dense backend on the host.
   allocate gives `bytes` bytes of device memory, NULL when there is no room
     or the device failed.  release gives them back; it is called once for
     every block from allocate, when no work is pending.
   upload and download copy `bytes` bytes between host and device memory and
     return when the copy is done.  The library calls them, as release, only
     when no work that it gave the device is pending: before its first
     dgemm or move, or after finish.
   dgemm is the product of the dense backend (above) on device memory.
   move sets, for r = 0 .. rows-1 and c = 0 .. (lengths ? lengths[r] :
     columns)-1, the element of target at the index t of (r, c) to
     alpha source[s(r, c)] (add 0) or adds that to it (add 1), in this
     order: the product first, then the sum.  lengths is NULL or an array in
     device memory, and no length exceeds columns.  The library never gives
     two pairs (r, c) of one call the same target element, and the target
     does not overlap the source.
   finish waits until all work given to the device is done.
   dgemm and move may return before their work is done: the device does the
   work in the order it was given.

   Every function except accept, allocate and release returns 0 on success.
   After a failure, or an allocation that gave NULL, the library stops using
   the device for the call: it calls finish if it gave the device work after
   the last finish, releases what it allocated, and computes the whole call
   with the dense backend from the unchanged k.  The library writes k only
   after the last download.  The results differ from those of the dense
   backend by the rounding of the products: the sums over the ABFs run in
   other orders.  Every element of k receives the parts of the centre atoms
   and of their V neighbours in the order of the host. */
typedef struct {
  void *context;
  int (*accept)(void *context, double multiplications, size_t bytes);
  void *(*allocate)(void *context, size_t bytes);
  void (*release)(void *context, void *pointer);
  int (*upload)(void *context, void *target, const void *source, size_t bytes);
  int (*download)(void *context, void *target, const void *source, size_t bytes);
  int (*dgemm)(void *context, char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda,
               const double *b, int ldb, double beta, double *c, int ldc);
  int (*move)(void *context, int rows, int columns, const int *lengths, double alpha, int add, double *target,
              const OpenMXExxDeviceIndex *target_index, const double *source,
              const OpenMXExxDeviceIndex *source_index);
  int (*finish)(void *context);
} OpenMXExxDevice;

/* ---- matrix products (section 7.2 of the guide) ------------------------- */

/* dgemm computes C = alpha op(A) op(B) + beta C for column-major arrays with
   the arguments of DGEMM of the BLAS (transa, transb: 'N' or 'T'), and
   returns 0 on success.  As DGEMM, it does not read C when beta is zero: the
   library passes arrays there that it has not set.  Whether it runs on the
   host or on a device is its own business; the arrays it receives are host
   memory.  context is passed through unchanged.  A table without dgemm
   (NULL) selects the loops of the library.

   parallel_for and threads are optional (NULL and 0: the library runs its
   loops on the calling thread), and so is device (NULL: no device).  The
   library reads all five members of a table, so a table is set as a whole,
   for example with "OpenMXExxDenseBackend table = {0};" before its members
   are assigned: a member that was never set is not NULL or 0.

   parallel_for calls body(argument, index) once for every index from 0 to
   count-1, returns when all calls have returned, and returns 0 on success.
   The calls may run at the same time on different threads.  The library
   hands it loops whose pieces write to different memory; dgemm is then
   called from those threads at the same time and must allow that.  threads
   is the largest number of pieces into which the library divides such a
   loop, the number of threads of parallel_for: count is at most threads,
   and smaller for a loop over fewer items.  With threads below 2 the
   library does not call parallel_for.  The library calls parallel_for from
   the thread that called the library, one loop at a time, and never from a
   body.

   The exchange build divides the ABFs of a centre atom into pieces, as many
   as threads or, with fewer ABFs, one for every ABF, and adds the sums over
   the ABFs piece by piece.  openmx_exx_add_gamma and openmx_exx_add_blocks
   also gather the rows of the density matrix that a centre atom needs, and
   add the last part of its contribution to k, in loops with as many pieces.
   These two steps give the same bits in pieces as on the calling thread:
   every element of k receives the same additions in the same order.  For
   one value of threads and a dgemm that gives the same result for the same
   arguments on every thread, the results do not depend on the order or the
   timing of the calls.  They can differ by rounding from the results for
   another value of threads; with one piece they are those of the loop on
   the calling thread.

   With a device (above) that accepts a call of openmx_exx_add_blocks or
   openmx_exx_add_gamma, the device computes the call, and dgemm and
   parallel_for are not used for it. */
typedef struct {
  int (*dgemm)(void *context, char transa, char transb, int m, int n, int k, double alpha,
               const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc);
  void *context;
  int (*parallel_for)(void *context, int count, void (*body)(void *argument, int index), void *argument);
  int threads;
  const OpenMXExxDevice *device;
} OpenMXExxDenseBackend;

/* The product that the library uses when no table is given: plain loops.
   context is ignored.  Returns 1 for arguments that DGEMM would reject. */
int openmx_exx_reference_dgemm(void *context, char transa, char transb, int m, int n, int k, double alpha,
                               const double *a, int lda, const double *b, int ldb, double beta, double *c,
                               int ldc);

/* ---- preparation for one structure (section 6.2) ------------------------ */

#define OPENMX_EXX_KERNEL_SHORT_RANGE 1       /* erfc(omega r)/r, HSE06; parameter: omega */
#define OPENMX_EXX_KERNEL_TRUNCATED_COULOMB 2 /* theta(R_c - r)/r, PBE0; parameter: R_c */

typedef struct {
  int kernel;                /* OPENMX_EXX_KERNEL_... */
  double kernel_parameter;   /* omega in 1/bohr, or R_c in bohr */
  double abf_threshold;      /* scf.exx.abf.pca */
  double v_cutoff;           /* scf.exx.v.cutoff, bohr */
  int radial_intervals;      /* 1DFFT.NumGridR */
  double reciprocal_maximum; /* square root of 1DFFT.EnergyCutoff, 1/bohr */
  int reciprocal_intervals;  /* 1DFFT.NumGridR */
} OpenMXExxOptions;

/* HSE06 with the defaults of the keywords: short-range kernel with omega =
   0.11, threshold 1e-4, cutoff 8.0, grids 900, 60.0, 900. */
void openmx_exx_default_options(OpenMXExxOptions *options);

/* The PAOs of one species as the main program holds them. */
typedef struct {
  double cutoff;          /* cutoff radius of the PAOs, bohr */
  int mesh_size;          /* number of points of the radial mesh */
  const double *mesh;     /* their radii, increasing */
  int shell_count;        /* number of radial functions, in the order of the main program */
  const int *angular;     /* angular momentum of every radial function */
  const double *values;   /* values[shell*mesh_size + point] */
} OpenMXExxSpecies;

typedef struct {
  double lattice[3][3];   /* lattice[i] is the lattice vector a_(i+1) */
  int atom_count;
  const double *positions; /* Cartesian, positions[3*atom + x] */
  const int *species;      /* species of every atom, 0 .. species_count-1 */
  int species_count;
  const OpenMXExxSpecies *species_data;
} OpenMXExxGeometry;

typedef struct OpenMXExxState OpenMXExxState;

/* A state without a structure; NULL when memory is exhausted. */
OpenMXExxState *openmx_exx_create(void);
/* NULL is allowed. */
void openmx_exx_destroy(OpenMXExxState *state);

/* ABFs, neighbour lists, T blocks and V blocks of the whole unit cell.  The
   tables of the species are kept between calls and made again only when the
   PAOs, the grids or the threshold change.  After a failure the state holds
   what it held before the call. */
int openmx_exx_prepare(OpenMXExxState *state, const OpenMXExxOptions *options,
                       const OpenMXExxGeometry *geometry, char *error, size_t capacity);

/* Sizes of the prepared structure.  -1 without a prepared structure or for
   an atom out of range. */
int openmx_exx_atom_count(const OpenMXExxState *state);
int openmx_exx_orbital_count(const OpenMXExxState *state);           /* N */
int openmx_exx_atom_orbitals(const OpenMXExxState *state, int atom); /* u_a */
int openmx_exx_atom_abfs(const OpenMXExxState *state, int atom);     /* n_P,a */
int openmx_exx_c_neighbours(const OpenMXExxState *state, int atom);
int openmx_exx_v_neighbours(const OpenMXExxState *state, int atom);

/* How equation (7) was solved during the last preparation. */
typedef struct {
  long systems;               /* linear systems solved */
  long eigen_decompositions;  /* of them, by eigen-decomposition */
  long dropped;               /* eigenvectors left out in total */
  double smallest_pivot;      /* of the scaled metrics factorized by Cholesky */
} OpenMXExxReport;
int openmx_exx_report(const OpenMXExxState *state, OpenMXExxReport *report);

/* ---- exchange matrix, Gamma point only (section 6.4) -------------------- */

/* Adds factor times the contributions of the centre atoms first_atom <= a <
   last_atom to k.  density and k are N x N, row-major; density is the density
   matrix of one spin channel (occupations between 0 and 1).  With k = 0 on
   entry, all atoms and factor = -alpha, k becomes the exchange matrix K of
   that spin channel, which is symmetric when the density matrix is.  The
   contributions of different atoms may be computed by different processes
   and added afterwards.  backend may be NULL for the built-in loops. */
int openmx_exx_add_gamma(const OpenMXExxState *state, int first_atom, int last_atom, double factor,
                         const double *density, double *k, const OpenMXExxDenseBackend *backend,
                         char *error, size_t capacity);

/* Equation (4) for one spin channel: (1/2) sum_ij D_ij K_ij over two arrays
   of count = N*N elements.  Returns 0 when an array is NULL. */
double openmx_exx_energy_gamma(size_t count, const double *density, const double *k);

/* Equation (15) for one spin channel: sum_ij D_ij (K_ij - K'_ij), K' the
   exchange matrix of the previous outer iteration.  Returns 0 when an array
   is NULL. */
double openmx_exx_change_gamma(size_t count, const double *density, const double *k, const double *k_previous);

/* ---- exchange matrix, site form for any k mesh (sections 6.3, 6.4) ------ */

/* The site form works with the blocks M_xy(R): rows on the orbitals of atom
   x in the cell at the origin, columns on those of atom y in the cell
   R = cell[0] a1 + cell[1] a2 + cell[2] a3, u_x x u_y elements, row-major.
   The blocks that a build reads from the density matrix and writes to the
   exchange matrix are those with a row site in loc(A) and a column site in
   S(A) for a centre atom A.  The state lists them once, ordered by x, y and
   the three components of the cell, and gives every block an offset in one
   array.  The order depends on the structure and the cutoffs only: all
   processes that prepare a structure get the same arrays.  K_xy(R) is zero
   for a block that is not listed.

   openmx_exx_prepare_blocks makes this list for the prepared structure; a
   new preparation discards it.  It does nothing when the list exists. */
int openmx_exx_prepare_blocks(OpenMXExxState *state, char *error, size_t capacity);

/* The number of blocks, -1 without a list, and the number of elements of an
   array that holds them all, 0 without a list. */
long openmx_exx_block_count(const OpenMXExxState *state);
size_t openmx_exx_block_elements(const OpenMXExxState *state);

/* Atoms, cell and offset of the block "index".  An output may be NULL.
   Returns 1 without a list or for an index out of range. */
int openmx_exx_block(const OpenMXExxState *state, long index, int *row_atom, int *column_atom, int cell[3],
                     size_t *offset);

/* The index of a block, or -1 when the list does not hold it. */
long openmx_exx_find_block(const OpenMXExxState *state, int row_atom, int column_atom, const int cell[3]);

/* openmx_exx_add_gamma for arrays of blocks: density holds the blocks
   D_xy(R) of one spin channel, and factor times the contributions of the
   centre atoms first_atom <= a < last_atom is added to k.  With k = 0 on
   entry, all atoms and factor = -alpha, k becomes K(R) of that spin channel.
   K_xy(R) is then the transpose of K_yx(-R), up to rounding, when the density
   matrix has that property.  Equations (4) and (15) are
   openmx_exx_energy_gamma and openmx_exx_change_gamma over the
   openmx_exx_block_elements() elements of the arrays. */
int openmx_exx_add_blocks(const OpenMXExxState *state, int first_atom, int last_atom, double factor,
                          const double *density, double *k, const OpenMXExxDenseBackend *backend,
                          char *error, size_t capacity);

/* Replaces every element of M_xy(R) and the element of M_yx(-R) that equals
   it in a matrix with M_xy(R) = M_yx(-R)^T by their mean.  The largest
   difference of such a pair before the call is written to asymmetry, which
   may be NULL. */
int openmx_exx_symmetrize_blocks(const OpenMXExxState *state, double *values, double *asymmetry,
                                 char *error, size_t capacity);

/* ---- derivative with respect to the positions (section 5.9) ------------- */

/* Adds to gradient[3*b + x] the part of the derivative of E_x with respect to
   the position of atom b along the axis x that goes through the T and V
   blocks of the centre atoms first_atom <= a < last_atom; gradient has three
   entries for every atom of the unit cell.  The density matrix is fixed and
   symmetric, K is factor times the contributions of all centre atoms
   (factor = -alpha for the exchange matrix), and E_x = (1/2) sum_ij D_ij K_ij
   for one spin channel.  With all atoms the entries are dE_x/dtau; the force
   of the exact exchange is their negative, and their sum over the atoms
   vanishes.  The parts of different centre atoms may be computed by
   different processes and added afterwards.  The neighbour lists of the
   prepared structure count as fixed, and the response of the orbitals is not
   included.  density is N x N, row-major, as for openmx_exx_add_gamma.  After
   a failure gradient is unchanged. */
int openmx_exx_add_position_gradient_gamma(const OpenMXExxState *state, int first_atom, int last_atom,
                                           double factor, const double *density, double *gradient,
                                           const OpenMXExxDenseBackend *backend, char *error, size_t capacity);

/* The same for arrays of blocks: density holds the blocks D_xy(R) of one
   spin channel with D_xy(R) = D_yx(-R)^T, as for openmx_exx_add_blocks, and
   E_x is openmx_exx_energy_gamma over the arrays of blocks. */
int openmx_exx_add_position_gradient_blocks(const OpenMXExxState *state, int first_atom, int last_atom,
                                            double factor, const double *density, double *gradient,
                                            const OpenMXExxDenseBackend *backend, char *error, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif
