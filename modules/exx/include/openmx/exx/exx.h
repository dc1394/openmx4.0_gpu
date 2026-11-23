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

/* ---- matrix products (section 7.2 of the guide) ------------------------- */

/* dgemm computes C = alpha op(A) op(B) + beta C for column-major arrays with
   the arguments of DGEMM of the BLAS (transa, transb: 'N' or 'T'), and
   returns 0 on success.  Whether it runs on the host or on a device is its
   own business; the arrays it receives are host memory.  context is passed
   through unchanged. */
typedef struct {
  int (*dgemm)(void *context, char transa, char transb, int m, int n, int k, double alpha,
               const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc);
  void *context;
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

#ifdef __cplusplus
}
#endif

#endif
