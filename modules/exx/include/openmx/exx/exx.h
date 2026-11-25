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
   and of their V neighbours in the order of the host.

   The weights of the forces (openmx_exx_add_block_weights_gamma_abfs and
   openmx_exx_add_block_weights_blocks_abfs) use the device in the same way:
   one accept for the call, and the arrays of all its centre atoms in device
   memory.  Their weights come back one centre atom (or part of one) at a
   time, two downloads each, and are added to the arrays of the caller at
   once.  After a failure the dense backend computes the centre atoms that
   did not come back; the others keep the weights of the device. */
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
   the calling thread.  The weights of the forces
   (openmx_exx_add_block_weights_gamma_abfs and
   openmx_exx_add_block_weights_blocks_abfs) divide the ABFs of a centre atom
   in the same way.  Nothing in them is a sum over the ABFs, so they give the
   same bits for every value of threads.

   With a device (above) that accepts a call of openmx_exx_add_blocks,
   openmx_exx_add_gamma or the weights of the forces, the device computes
   the call, and dgemm and parallel_for are not used for it. */
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

/* The share "part" of "parts" of that preparation, for processes that divide
   it: the ABFs, the neighbour lists and the sizes are those of
   openmx_exx_prepare on every process, and the T and V blocks of the pairs of
   atoms are computed for one share of the pairs and zero for the others.
   Every pair belongs to exactly one share, so the element-wise sums of the
   blocks over the shares are the blocks of openmx_exx_prepare.  With several
   parts the caller adds the blocks of every atom over the shares
   (openmx_exx_get_atom_tables) and stores the sums in every state
   (openmx_exx_set_atom_tables); until that is done for all atoms the builds
   and the derivatives fail.  openmx_exx_report counts the systems of the
   share.  parts = 1 is openmx_exx_prepare. */
int openmx_exx_prepare_part(OpenMXExxState *state, const OpenMXExxOptions *options,
                            const OpenMXExxGeometry *geometry, int part, int parts, char *error,
                            size_t capacity);

/* openmx_exx_prepare_part with the pairs of the share divided among the
   pieces of backend: as many pieces as its threads when it has a loop, and
   pair k in piece k mod pieces.  Only the loop of backend is used, not its
   product or its device.  A pair writes its own blocks, so the blocks and
   the report are those of openmx_exx_prepare_part, bit for bit, for every
   number of pieces.  backend NULL is openmx_exx_prepare_part. */
int openmx_exx_prepare_part_pieces(OpenMXExxState *state, const OpenMXExxOptions *options,
                                   const OpenMXExxGeometry *geometry, int part, int parts,
                                   const OpenMXExxDenseBackend *backend, char *error, size_t capacity);

/* The T blocks and the V blocks of one atom as one array: its number of
   elements (0 without a prepared structure or for an atom out of range), a
   copy of the elements, and their replacement.  The order of the elements
   depends on the structure and the cutoffs only.  The last two return 1
   without a prepared structure, for an atom out of range or for a NULL
   array. */
size_t openmx_exx_atom_table_elements(const OpenMXExxState *state, int atom);
int openmx_exx_get_atom_tables(const OpenMXExxState *state, int atom, double *values);
int openmx_exx_set_atom_tables(OpenMXExxState *state, int atom, const double *values);

/* Sizes of the prepared structure.  -1 without a prepared structure or for
   an atom out of range. */
int openmx_exx_atom_count(const OpenMXExxState *state);
int openmx_exx_orbital_count(const OpenMXExxState *state);           /* N */
int openmx_exx_atom_orbitals(const OpenMXExxState *state, int atom); /* u_a */
int openmx_exx_atom_abfs(const OpenMXExxState *state, int atom);     /* n_P,a */
int openmx_exx_c_neighbours(const OpenMXExxState *state, int atom);
int openmx_exx_v_neighbours(const OpenMXExxState *state, int atom);
/* The V neighbours of the preparation: those of openmx_exx_v_neighbours
   before a folding, and their number before the folding after one. */
int openmx_exx_prepared_v_neighbours(const OpenMXExxState *state, int atom);

/* Folds the V neighbours of the prepared structure by the period of a k
   mesh of period[0] x period[1] x period[2] points (section 6.4;
   fold_v_neighbours() of system.hpp): the V neighbours of an atom whose
   cells agree modulo the period become one site, with the sum of their V
   blocks, so that the builds of the site form go through one V neighbour
   per class.  For a density matrix of the points of a regular mesh of
   that period, shifted by half a point or not, K(k) at the points of the
   mesh, the energy and the derivatives keep their values within rounding;
   the blocks K(R), and K(k) away from the points of the mesh, change.  The
   blocks of a divided preparation must have been added up first
   (openmx_exx_set_atom_tables).  The call discards the list of the blocks
   of the site form (openmx_exx_prepare_blocks makes it again), and the
   tables of the atoms (openmx_exx_get_atom_tables) hold the folded blocks
   afterwards.  The derivative with respect to the positions goes through
   the pairs of sites of the preparation.  A new preparation discards the
   folding.  Returns 1 for a NULL period, a period below 1 and a structure
   that is folded already, and leaves the state as it was. */
int openmx_exx_fold_v_neighbours(OpenMXExxState *state, const int period[3], char *error, size_t capacity);

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

/* The contribution of a centre atom is a sum over its ABFs, and a build may
   take a part of them.  The ABFs of the unit cell are numbered atom after
   atom, in the order of the ABFs of every atom: openmx_exx_abf_count is
   their number (-1 without a prepared structure), and
   openmx_exx_add_gamma_abfs adds factor times the contributions of the ABFs
   first_abf <= P < last_abf, which may end inside an atom.  The ranges of a
   division of all ABFs give K when added, up to rounding; a range of whole
   atoms gives what openmx_exx_add_gamma gives for those atoms.  An empty
   range adds nothing. */
long openmx_exx_abf_count(const OpenMXExxState *state);
int openmx_exx_add_gamma_abfs(const OpenMXExxState *state, long first_abf, long last_abf, double factor,
                              const double *density, double *k, const OpenMXExxDenseBackend *backend,
                              char *error, size_t capacity);

/* A division of the ABFs among processes: process r takes the ABFs
   boundaries[r] <= P < boundaries[r+1], counted as for
   openmx_exx_add_gamma_abfs, and boundaries gets processes + 1 numbers.  Of
   all divisions into contiguous ranges, it has the smallest largest
   estimate of the time of a process, in the site form when sites is not
   zero and in the Gamma-point form otherwise, for the half build (section
   6.4) when half is not zero.  The estimate weighs the products of every
   ABF by their speed and adds, for every atom whose ABFs a range holds, the
   work that each part of a centre atom repeats.  Processes at the end may
   get empty ranges.  The division depends on the structure and the cutoffs
   only, so every process computes the same one.  Returns 1 without a
   prepared structure, for a NULL array or for processes < 1. */
int openmx_exx_divide_abfs(const OpenMXExxState *state, int sites, int half, int processes, long *boundaries);

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

/* openmx_exx_add_gamma_abfs for arrays of blocks. */
int openmx_exx_add_blocks_abfs(const OpenMXExxState *state, long first_abf, long last_abf, double factor,
                               const double *density, double *k, const OpenMXExxDenseBackend *backend,
                               char *error, size_t capacity);

/* Replaces every element of M_xy(R) and the element of M_yx(-R) that equals
   it in a matrix with M_xy(R) = M_yx(-R)^T by their mean.  The largest
   difference of such a pair before the call is written to asymmetry, which
   may be NULL. */
int openmx_exx_symmetrize_blocks(const OpenMXExxState *state, double *values, double *asymmetry,
                                 char *error, size_t capacity);

/* ---- the half build (section 6.4) ---------------------------------------- */

/* K is symmetric when the density matrix is, and the contribution of a
   centre atom a through its V neighbour (b, R) is the transpose of that of
   the centre b through its V neighbour (a, -R).  The half build takes one
   pair of the two: in the site form the V neighbours with R > 0 in the order
   of the components, or with R = 0 and b > a, and the centre itself with
   half its V block; in the Gamma form, where the images of an atom are one
   group, the atoms b > a and half the group of the centre itself.  It takes
   about half the products of the V neighbours of a build.
   openmx_exx_add_gamma_half_abfs and openmx_exx_add_blocks_half_abfs add
   factor times the half build of the ABFs first_abf <= P < last_abf, with
   the arguments of openmx_exx_add_gamma_abfs and openmx_exx_add_blocks_abfs.
   The ranges of a division of all ABFs add up to X, and
   openmx_exx_mirror_gamma or openmx_exx_mirror_blocks then turns X into
   K = X + X^T, which equals K of the whole build within rounding and is
   symmetric to the last bit.  The density matrix must be symmetric, with
   D_xy(R) = D_yx(-R)^T in the site form.  openmx_exx_divide_abfs with half
   divides its ABFs among processes. */
int openmx_exx_add_gamma_half_abfs(const OpenMXExxState *state, long first_abf, long last_abf, double factor,
                                   const double *density, double *k, const OpenMXExxDenseBackend *backend,
                                   char *error, size_t capacity);
int openmx_exx_add_blocks_half_abfs(const OpenMXExxState *state, long first_abf, long last_abf, double factor,
                                    const double *density, double *k, const OpenMXExxDenseBackend *backend,
                                    char *error, size_t capacity);

/* M + M^T of an N x N array, and M_xy(R) + M_yx(-R)^T of an array of blocks,
   written over the array. */
int openmx_exx_mirror_gamma(const OpenMXExxState *state, double *values, char *error, size_t capacity);
int openmx_exx_mirror_blocks(const OpenMXExxState *state, double *values, char *error, size_t capacity);

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

/* ---- the same derivative for processes that divide it (section 6.7) ----- */

/* E_x depends on the positions through the T and V blocks.  The derivative
   takes two steps.  The first one makes the weights of the blocks, the
   derivatives of E_x with respect to their elements, from the density
   matrix.  The weights are separate for every ABF of the centre atoms, as
   the contributions to K are, and the ABFs are numbered as for
   openmx_exx_add_gamma_abfs: these calls add the weights of the ABFs
   first_abf <= P < last_abf to weights[a], which has
   openmx_exx_atom_table_elements(a) numbers in the order of
   openmx_exx_get_atom_tables.  A range may end inside an atom; it then fills
   the rows of its ABFs in the arrays of that atom.  weights has an entry for
   every atom of the unit cell; the entries of the atoms that the range does
   not touch are not read.  factor and density are those of the calls above,
   and the weights of several spin channels add up.  Every number of the
   arrays comes from one ABF, so processes that divide the ABFs add their
   arrays up without rounding.  An empty range adds nothing.  A call that
   fails may have added the weights of some of its atoms.  A device in the
   backend computes the call when it accepts it (above); its weights differ
   from those of the host by rounding. */
int openmx_exx_add_block_weights_gamma_abfs(const OpenMXExxState *state, long first_abf, long last_abf, double factor,
                                            const double *density, double *const *weights,
                                            const OpenMXExxDenseBackend *backend, char *error, size_t capacity);
int openmx_exx_add_block_weights_blocks_abfs(const OpenMXExxState *state, long first_abf, long last_abf,
                                             double factor, const double *density, double *const *weights,
                                             const OpenMXExxDenseBackend *backend, char *error, size_t capacity);

/* The same weights in the form of the half build (section 6.7), for the
   E_x of openmx_exx_add_gamma_half_abfs and openmx_exx_add_blocks_half_abfs.
   A centre goes through the V neighbours of the half build only: the V
   block of a pair of sites gets the derivative of the pair once, and the T
   blocks of an atom get the part of the derivative that goes through the
   products with the V blocks of other centres.  The weights of an atom
   therefore differ from those above, but the derivative of
   openmx_exx_add_position_gradient_pairs is the same within rounding.  A
   range adds to the arrays of its atoms and of the atoms of their V
   neighbours: weights needs an array for every atom of the unit cell unless
   the range is empty.  Processes that divide the ABFs add their arrays up,
   now with rounding.  A range takes about 0.6 times the time of the calls
   above: half of the V neighbours, with three large products each instead
   of two.  factor, density, the device and the failures are as above. */
int openmx_exx_add_block_weights_gamma_half_abfs(const OpenMXExxState *state, long first_abf, long last_abf,
                                                 double factor, const double *density, double *const *weights,
                                                 const OpenMXExxDenseBackend *backend, char *error, size_t capacity);
int openmx_exx_add_block_weights_blocks_half_abfs(const OpenMXExxState *state, long first_abf, long last_abf,
                                                  double factor, const double *density, double *const *weights,
                                                  const OpenMXExxDenseBackend *backend, char *error, size_t capacity);

/* The second step goes through the pairs of sites: it adds to gradient the
   part of dE_x/dtau of the share "part" of "parts" of the pairs, from the
   weights of all atoms (weights[a] for every atom of the unit cell, complete
   on the process that calls).  Every pair of two atoms is visited once, and
   the shares of a division add up to the derivative that
   openmx_exx_add_position_gradient_gamma or
   openmx_exx_add_position_gradient_blocks gives for all centre atoms, within
   rounding.  The density matrix is not needed any more.  After a failure
   gradient is unchanged. */
int openmx_exx_add_position_gradient_pairs(const OpenMXExxState *state, const double *const *weights, int part,
                                           int parts, double *gradient, char *error, size_t capacity);

/* The same with the pairs of the share divided among the pieces of backend,
   as in openmx_exx_prepare_part_pieces.  The vector of every pair is kept,
   and the vectors are added in the order of the pairs, so the derivative is
   that of openmx_exx_add_position_gradient_pairs, bit for bit, for every
   number of pieces.  backend NULL is openmx_exx_add_position_gradient_pairs. */
int openmx_exx_add_position_gradient_pairs_pieces(const OpenMXExxState *state, const double *const *weights,
                                                  int part, int parts, const OpenMXExxDenseBackend *backend,
                                                  double *gradient, char *error, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif
