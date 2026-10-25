#ifndef OPENMX_CORE_VNA_SIMD_H
#define OPENMX_CORE_VNA_SIMD_H
#include "openmx_core_grid.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Host kernels of Set_ProExpn_VNA (the neutral-atom projector expansion) for
   the CPU path. Every kernel keeps the legacy per-element operation order,
   so results are bitwise those of the scalar loops when both are compiled
   without contraction; SIMD lanes span independent outputs only. Each entry
   returns 0 for an unsupported mode or an invalid description, with nothing
   written, else 1. Pure, re-entrant and allocation-free. */

/* Gauss-Legendre contraction of the DS_VNA assembly: for m < count and
   r < rows, in ascending i,
     out0[m*out_stride+r] = sum_i w0[m][i] * table[i*table_stride+r]
     out1[m*out_stride+r] = sum_i w1[m][i] * table[i*table_stride+r]
   where w0/w1 are the legacy Bes00/Bes01 rows (radial function times Bessel
   value) and table is the projector table transposed to [point][projector]. */
typedef struct {
  int points, rows, count;
  size_t table_stride, out_stride;
  const double *table;
  const double *const *w0, *const *w1;
  double *out0, *out1;
} OpenMXVnaContract;
int openmx_core_vna_contract(const OpenMXVnaContract*, int mode);

/* Spherical_Bessel2 of Set_ProExpn_VNA for a batch of arguments x[i]:
   sb[l*stride+i] = j_l(x[i]) and dsb[l*stride+i] = j_l'(x[i]) for l <= lmax
   (lmax <= 30). When transposed/transposed_derivative are set, the same
   values also go to transposed[i*transposed_stride+l]. Each lane follows the
   scalar routine's own recurrence length and rescaling decisions. */
typedef struct {
  int points, lmax;
  const double *x;
  double *sb, *dsb;
  size_t stride;
  double *transposed, *transposed_derivative;
  size_t transposed_stride;
} OpenMXVnaBessel;
int openmx_core_vna_bessel(const OpenMXVnaBessel*, int mode);

/* HVNA23 quadrature with lanes across the Bessel orders l < orders:
     t = ((((scale*crude[i])*product[i][l])*weights[i])*k[i])*k[i]
     sum[l] += t*bessel[i][l];  sum_derivative[l] += (t*bessel_derivative[i][l])*k[i]
   in ascending i, where the three tables are [point][table_stride]. */
typedef struct {
  int points, orders;
  size_t table_stride;
  const double *product, *bessel, *bessel_derivative;
  const double *crude, *weights, *wavevectors;
  double scale;
  double *sum, *sum_derivative;
} OpenMXVnaQuadrature;
int openmx_core_vna_quadrature(const OpenMXVnaQuadrature*, int mode);

/* Energy-weighted projector contraction of the HVNA host path, for every
   m < rows and n < columns:
     sum = 0; L = 0;
     for each group: tmp = 0; for each member: tmp += product(left[m][L], right[n][L]); ++L;
                     sum += energies[group]*tmp;
     first ? matrix[m*stride+n] = sum : matrix[m*stride+n] += sum
   product is (double)(a*b) in float when fp32 is set, else (double)a*(double)b.
   One operand block is transposed to [component][...]: with right_transposed
   the right block is right[L*right_stride+n] and the left block keeps its
   legacy rows left[m*left_stride+L]; otherwise left[L*left_stride+m] and
   right[n*right_stride+L]. */
typedef struct {
  int rows, columns, groups, fp32, first, right_transposed;
  const int *lengths;
  const double *energies;
  const float *left;
  size_t left_stride;
  const float *right;
  size_t right_stride;
  double *matrix;
  size_t stride;
} OpenMXVnaHvnaBlock;
int openmx_core_vna_hvna_block(const OpenMXVnaHvnaBlock*, int mode);

#ifdef __cplusplus
}
#endif
#endif
