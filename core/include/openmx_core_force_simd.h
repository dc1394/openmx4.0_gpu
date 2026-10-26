#ifndef OPENMX_CORE_FORCE_SIMD_H
#define OPENMX_CORE_FORCE_SIMD_H
#include "openmx_core_grid.h"
#include "openmx_core_projector_force.h"
#include "openmx_core_force_vna_simd.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Host kernels of the force stages (Force4B/dHVNA, Force3, Force_HNL) for the
   CPU fallbacks. Every kernel keeps the legacy per-element arithmetic and
   summation order: SIMD lanes span independent matrix columns or rows, never a
   reduction. mode is an OPENMX_GRID_* host SIMD mode; every entry returns 0
   and writes nothing for an unsupported mode, a missing buffer, an invalid
   shape or a scratch area smaller than the matching *_scratch() size. The
   kernels do not allocate; scratch is caller-owned and thread-private. */

/* Force #3 overlap-grid contraction of one atom pair (Force3 part 2):
   for every point p, spin s:
     tmp0[i] = sum_j (double)neighbor_rows[...][j] * density[s][i][j]
     tmp_a   = sum_i derivative[center[p]][i][a] * tmp0[i]      (a = x,y,z)
     sums[a] += tmp_a * potential[s][center[p]]
   accumulated into sums in exactly that order (points, spins, i ascending).
   neighbor selects the orbital row of point p (NULL: row p). */
typedef struct {
  int rows, columns, spins, points;
  const double *const *density[4];           /* density[s][i] -> columns doubles */
  const int *center;                         /* [points] */
  const int *neighbor;                       /* [points] or NULL */
  const float *const *neighbor_rows;         /* rows of >= columns floats */
  const double *const *const *derivative;    /* derivative[center][i] -> 3 doubles */
  const double *const *potential;            /* potential[s][center] */
} OpenMXForceGridPair;
size_t openmx_core_force_grid_scratch(int rows,int columns,int spins);
int openmx_core_force_grid_pair(const OpenMXForceGridPair*,double sums[3],void *scratch,size_t scratch_bytes,int mode);

/* Point-major form of the same contraction for all pairs of one atom. The
   visits of the atom's grid points are listed point by point (CSR:
   visits [point_offsets[p], point_offsets[p+1]) belong to point p), each
   naming its pair and its orbital row; the pair's density matrix comes as
   prepared planes dm[(pair*spins + s)*columns_max*pad_rows + j*pad_rows + i]
   (openmx_core_force_grid_density; pad_rows = rows rounded up to
   OPENMX_FORCE_SIMD_PAD). For visit v of point p and spin s the kernel forms
   tmp0, tmp_a exactly as above and stores
     terms[(visit_term[v]*spins + s)*3 + a] = tmp_a * potential[s][p]
   so that the caller adds the terms in the legacy pair/point/spin order. The
   derivative of point p is read flat at derivative + p*point_stride +
   i*row_stride + a. Pure, re-entrant, allocation-free; callers split the
   points across threads. */
typedef struct {
  int rows, spins, points, columns_max;
  size_t pad_rows;
  const double *dm;                          /* prepared density planes of every pair */
  const int *pair_columns;                   /* [pairs]: NO1 of each pair */
  const size_t *point_offsets;               /* [points+1] */
  const int *visit_pair, *visit_row;         /* [visits] */
  const int *visit_term;                     /* [visits]: term slot (legacy order) */
  const float *const *const *pair_rows;      /* pair_rows[pair][visit_row] -> columns floats */
  const double *derivative;                  /* flat derivatives of the points */
  size_t point_stride, row_stride;
  const double *const *potential;            /* potential[s][point] */
  double *terms;                             /* [terms*spins*3] */
} OpenMXForceGridPoints;
/* dm plane bytes for pairs pairs: pairs*spins*columns_max*pad_rows doubles. */
size_t openmx_core_force_grid_density_bytes(int rows,int columns_max,int spins,int pairs);
/* Transposes density[s][i][j] (columns of this pair) into the planes of pair
   index pair; padding rows are zero. */
int openmx_core_force_grid_density(const double *const *density[4],int rows,int columns,int columns_max,int spins,
    int pair,double *dm,size_t dm_bytes,int mode);
/* Visits of the points [first, first+count) of the atom. */
int openmx_core_force_grid_points(const OpenMXForceGridPoints*,int first,int count,void *scratch,size_t scratch_bytes,int mode);
size_t openmx_core_force_grid_points_scratch(int rows);

/* Collinear nonlocal projector block: the same contract as
   openmx_core_projector_force_block (terms, transpose, value_first, sign,
   reverse_energy, accumulate) for a species whose view is not relativistic;
   a relativistic species is refused (return 0) so callers keep the Core's
   general block. The per-element products and the channel/magnetic order of
   the scalar Core block are preserved. */
size_t openmx_core_force_nl_scratch(int columns,size_t components,size_t terms);
int openmx_core_force_nl_block(const OpenMXForceProjectorSpecies*,const OpenMXProjectorForceBlock*,
    void *scratch,size_t scratch_bytes,int mode);

#ifdef __cplusplus
}
#endif
#endif
