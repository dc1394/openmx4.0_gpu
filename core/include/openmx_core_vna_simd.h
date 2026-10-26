#ifndef OPENMX_CORE_VNA_SIMD_H
#define OPENMX_CORE_VNA_SIMD_H
#include "openmx_core_grid.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Host kernels of Set_ProExpn_VNA (the neutral-atom projector expansion) and
   of the two-centre radial quadratures of Set_OLP_Kin and Set_Nonlocal for
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
   right[n*right_stride+L]. Without fp32, left_double and right_double may
   supply both blocks already widened to double (same layouts and strides;
   left and right are then not read): the products and the results are the
   same, with no conversion in the inner loop. Zero-initialize them otherwise. */
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
  const double *left_double, *right_double;
} OpenMXVnaHvnaBlock;
int openmx_core_vna_hvna_block(const OpenMXVnaHvnaBlock*, int mode);

/* The m sum of the DS_VNA assembly for one (L0, Mul0, L) block over its
   range of Bessel orders LL = ll_first .. ll_first+ll_count-1 (ascending, as
   the legacy LL loop): starting from zero, every pair (M0, M1) receives for
   each LL with |M0-M1| <= LL, in the legacy operation order,
     S  += Cmul(Cpow(LL), Y(LL,m))   * (gaunt(LL)[M0][M1]*s0(LL))   (real and imaginary parts)
     R  += Cmul(Cpow(LL), Y(LL,m))   * (gaunt(LL)[M0][M1]*s1(LL))
     T  += Cmul(Cpow(LL), dYt(LL,m)) * (gaunt(LL)[M0][M1]*s0(LL))
     P  += Cmul(Cpow(LL), dYp(LL,m)) * (gaunt(LL)[M0][M1]*s0(LL))
   with m = M0-M1, and every element of the block is written (zero when no
   LL contributes). The harmonics come reversed and zero padded per LL:
   cy[c][LL*cy_stride + pad + LL - m] for c = Y.r, Y.i, dYt.r, dYt.i, dYp.r,
   dYp.i, with pad >= l0+l1 and cy_stride >= pad + max(2*LLmax+1,
   LLmax+l0+l1+8); cy_length counts the entries of each table. cpow_r/cpow_i,
   s0, s1 and gaunt are per LL of the range (ll_count entries); gaunt[t] is the
   row of (L0, M0 = -l0, LL, L1), rows [rows][gaunt_stride] with M1 ascending.
   The eight planes acc (S.r, S.i, R.r, R.i, T.r, T.i, P.r, P.i) are
   [rows][acc_stride] at the block origin (M0 = -l0, M1 = -l1). */
typedef struct {
  int rows, columns, l0, l1, pad, ll_first, ll_count;
  size_t cy_length, cy_stride;
  const double *cy[6];
  const double *cpow_r, *cpow_i, *s0, *s1;
  const double *const *gaunt;
  size_t gaunt_stride;
  double *acc[8];
  size_t acc_stride;
} OpenMXVnaAngular;
int openmx_core_vna_angular(const OpenMXVnaAngular*, int mode);

/* The two complex-to-real conversions of one (L0, Mul0, L1, Mul1) block and
   the DS_VNA rows they produce, in the legacy order: first
     C[q][M0][M1] = sum_k Cmul(conj(conv0[M0][k]), acc_q[k][M1])      (k = -l0..l0)
   then the real part of sum_k C[q][M0][k]*convt[k][M1] (k = -l1..l1, the
   legacy adds (a.r*b.r - a.i*b.i) per k), and
     out[0] = 8*(float)S,  out[1..3] = -8*(float)(derivative combinations)
   with the coefficients c1..c8 = siT*coP, coT*coP/r, siP/siT/r, siT*siP,
   coT*siP/r, coP/siT/r, coT, siT/r; polar drops the c3/c6 terms and onsite
   writes zero derivatives. conv0 is the interleaved complex conversion
   matrix of l0 (its element (l0+M0, l0+k), 0-based indices, at
   conv0[2*((l0+M0)*width+(l0+k))]); convt_r/convt_i the transposed matrix of
   l1 (element (l1+k, l1+M1) at [(l1+k)*ld+(l1+M1)]); out rows are float,
   [rows0][out_stride] at (M0 = -l0, M1 = -l1). rows0 <= 16, columns <= 32. */
typedef struct {
  int rows0, columns, onsite, polar;
  const double *acc[8];
  size_t acc_stride;
  const double *conv0;
  size_t width;
  const double *convt_r, *convt_i;
  size_t ld;
  float *out[4];
  size_t out_stride;
  double c1, c2, c3, c4, c5, c6, c7, c8;
} OpenMXVnaConvert;
int openmx_core_vna_convert(const OpenMXVnaConvert*, int mode);

/* The radial quadrature of Set_OLP_Kin for one pair, lanes across the radial
   function pairs (combos) and every order l < orders sharing each loaded row;
   per element the legacy order and association:
     k2 = k*k; coe0 = 0.5 at the first and last point, else 1;
     t0 = ((coe0*h)*k2)*rf0[i][c];  t1 = t0*sb[l][i];  t2 = (t0*k)*sbp[l][i];
     t3 = t1*rf1[i][c];  t4 = t2*rf1[i][c];
     S[l][c] += t3;  K[l][c] += t3*k2;  Sr[l][c] += t4;  Kr[l][c] += t4*k2
   in ascending i, with Sr and Kr zero for an on-site pair. rf0t/rf1t are the
   two species' radial rows transposed to [point][table_stride] (combo c in
   column c); sb/sbp are [orders][sb_stride]; out[0..3] = S, K, Sr, Kr as
   [orders][out_stride]. */
typedef struct {
  int points, orders, combos, onsite;
  double h;
  const double *k;
  const double *rf0t, *rf1t;
  size_t table_stride;
  const double *sb, *sbp;
  size_t sb_stride;
  double *out[4];
  size_t out_stride;
} OpenMXVnaOverlapRadial;
int openmx_core_vna_overlap_radial(const OpenMXVnaOverlapRadial*, int mode);

/* The radial quadrature of Set_Nonlocal for one pair and spin-orbit index,
   lanes across the (basis function, projector) pairs (combos) and every
   order l < orders sharing each loaded row; per element the legacy order and
   association:
     coe0 = 0.5 at the first and last point, else 1;
     t0 = (((coe0*h)*k)*k)*rf[i][c];  t1 = t0*sb[l][i];  t2 = (t0*k)*sbp[l][i];
     S[l][c] += t1*nlrf[i][c];  Sr[l][c] += t2*nlrf[i][c]
   in ascending i, with Sr zero for an on-site pair. rft/nlrft are the basis
   and projector rows of each combo transposed to [point][table_stride];
   sb/sbp are [orders][sb_stride]; out[0..1] = S, Sr as [orders][out_stride]. */
typedef struct {
  int points, orders, combos, onsite;
  double h;
  const double *k;
  const double *rft, *nlrft;
  size_t table_stride;
  const double *sb, *sbp;
  size_t sb_stride;
  double *out[2];
  size_t out_stride;
} OpenMXVnaNonlocalRadial;
int openmx_core_vna_nonlocal_radial(const OpenMXVnaNonlocalRadial*, int mode);

#ifdef __cplusplus
}
#endif
#endif
