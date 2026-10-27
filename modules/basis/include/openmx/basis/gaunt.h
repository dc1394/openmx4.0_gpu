#ifndef OPENMX_BASIS_GAUNT_H
#define OPENMX_BASIS_GAUNT_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Immutable coefficients G(l0,m0,l1,m1,LL,m0-m1). Layout:
   [l0*(l0+1)+m0][LL][l1][l1+m1], with padded final width 2*l1max+1.
   The producer owns storage; no pointer trees or application initialization. */
typedef struct {
  int l0max, l1max, llmax;
  size_t row_stride;
  const double *values;
} OpenMXGauntTableData;

/* Unchecked row access for validated active indices. */
static inline size_t openmx_gaunt_table_row_offset(const OpenMXGauntTableData *table,
    int l0, int m0, int ll, int l1)
{
  return (((size_t)(l0*(l0+1)+m0)*(size_t)(table->llmax+1)+(size_t)ll)*
      (size_t)(table->l1max+1)+(size_t)l1)*table->row_stride;
}

/* Integral of conj(Y_lm)*Y_l1m1*Y_l2m2 over solid angle. Normalized complex
   harmonics use the Condon-Shortley phase. All l are 0..38 and |m|<=l.
   Allowed triangle cases require sum(l)<=38; other degrees represent zeros.
   Forbidden triangle, parity and magnetic sums give zero to the
   accuracy of the inherited factorial sum. Half-integer spin is not supported.
   Returns 1 on success, 0 on error; preserves numerical output on failure.
   No allocation on success. Diagnostic and numerical storage must not overlap. */
int openmx_basis_gaunt(int l, int m, int l1, int m1, int l2, int m2,
    double *output, char *error, size_t error_capacity);

/* Fill a table in caller storage; capacity counts doubles. All maxima are
   0..38, and every allowed triangle in the table must have degree sum <=38.
   Rectangular selection-zero rows may exceed that sum. Padding is zero;
   trailing capacity is untouched.
   On success out borrows values; on failure both arrays/descriptor are unchanged.
   No allocation on success; error storage must not overlap either output. */
int openmx_basis_gaunt_table_fill(int l0max, int l1max, int llmax,
    double *values, size_t capacity, OpenMXGauntTableData *out,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
