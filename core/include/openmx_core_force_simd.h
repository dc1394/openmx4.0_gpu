#ifndef OPENMX_CORE_FORCE_SIMD_H
#define OPENMX_CORE_FORCE_SIMD_H
#include "openmx_core_grid.h"
#include "openmx_core_projector_force.h"
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

#define OPENMX_FORCE_SIMD_PAD 16

/* A lane operand prepared once and shared by several block calls: the rows of
   each component transposed to a plane [projector][pad] of floats, pad = count
   rounded up to OPENMX_FORCE_SIMD_PAD, planes consecutive per component.
   openmx_core_force_vna_lanes fills caller memory of at least
   openmx_core_force_vna_lanes_bytes() bytes from rows[c][n] (row n of
   component c, projectors floats each) and describes it in *lanes. */
typedef struct {
  const float *planes;
  int count, projectors, components;
  size_t pad;
} OpenMXForceVnaLanes;
size_t openmx_core_force_vna_lanes_bytes(int count,int projectors,int components);
int openmx_core_force_vna_lanes(const float *const *const *rows,int components,int count,int projectors,
    void *memory,size_t bytes,OpenMXForceVnaLanes *lanes,int mode);

/* Neutral-atom projector derivative block of the fused Force4B traces:
     terms 1:  H_c[m][n] =   sum_l  derivative[c][m][l] * value[n][l]
     terms 2:  H_c[m][n] = - sum_l (derivative[c][m][l] * value[n][l]
                                   + row_value[m][l] * column_derivative[c][n][l])
   for c = x,y,z (the two products of a term-2 step are added first, then
   subtracted, as in Force4B_case2_trace_fused). fp32 != 0 rounds each float
   product in float before widening, else both operands widen first. Rows are
   float projector rows of at least projectors entries.
   lanes_rows selects the operand the SIMD lanes span: 0 = the columns n (the
   value side: planes value, column_derivative x,y,z), 1 = the rows m (the
   derivative side: planes derivative x,y,z, row_value); the other operand is
   streamed row by row. The lane operand is read from *lanes when given (count
   = that side's extent, same projectors, enough components), else transposed
   into scratch of openmx_core_force_vna_scratch(lane count, projectors, terms,
   lanes_rows) bytes. output[c] receives H_c at [m*output_stride + n] for
   lanes_rows 0 and at [n*output_stride + m] for lanes_rows 1 (the stride is
   at least the lane count). */
typedef struct {
  int rows, columns, projectors, terms, fp32, lanes_rows;
  const float *const *derivative[3];        /* [rows] */
  const float *const *row_value;            /* [rows], terms 2 */
  const float *const *value;                /* [columns] */
  const float *const *column_derivative[3]; /* [columns], terms 2 */
  const OpenMXForceVnaLanes *lanes;         /* optional prepared lane operand */
  double *output[3];
  size_t output_stride;
} OpenMXForceVnaBlock;
size_t openmx_core_force_vna_scratch(int lane_count,int projectors,int terms,int lanes_rows);
int openmx_core_force_vna_block(const OpenMXForceVnaBlock*,void *scratch,size_t scratch_bytes,int mode);

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
