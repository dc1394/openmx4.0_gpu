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

/* Neutral-atom projector derivative block of the fused Force4B traces:
     terms 1:  H_c[m][n] =   sum_l  derivative[c][m][l] * value[n][l]
     terms 2:  H_c[m][n] = - sum_l (derivative[c][m][l] * value[n][l]
                                   + row_value[m][l] * column_derivative[c][n][l])
   for c = x,y,z (the two products of a term-2 step are added first, then
   subtracted, as in Force4B_case2_trace_fused). fp32 != 0 rounds each float
   product in float before widening, else both operands widen first. Rows are
   float projector rows of at least projectors entries; output[c] receives
   rows x columns doubles with row stride output_stride (>= columns). */
typedef struct {
  int rows, columns, projectors, terms, fp32;
  const float *const *derivative[3];        /* [rows] */
  const float *const *row_value;            /* [rows], terms 2 */
  const float *const *value;                /* [columns] */
  const float *const *column_derivative[3]; /* [columns], terms 2 */
  double *output[3];
  size_t output_stride;
} OpenMXForceVnaBlock;
size_t openmx_core_force_vna_scratch(int columns,int projectors,int terms);
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
