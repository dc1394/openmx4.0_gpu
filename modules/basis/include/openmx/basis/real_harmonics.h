#ifndef OPENMX_BASIS_REAL_HARMONICS_H
#define OPENMX_BASIS_REAL_HARMONICS_H
#include "complex.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Immutable [degree][real row][complex column] coefficients. Columns are
   m=-degree,...,+degree; rows use the OpenMX real-orbital order. All degrees
   share width >= 2*degree+1. Storage and its lifetime belong to the producer. */
typedef struct {
  size_t orders, width;
  const dcomplex *coefficients;
} OpenMXRealHarmonicData;

/* Unchecked access for a validated degree and active row/column. */
static inline dcomplex openmx_real_harmonic_coefficient(
    const OpenMXRealHarmonicData *table, size_t degree, size_t row, size_t column)
{
  return table->coefficients[(degree*table->width+row)*table->width+column];
}

/* Fill (maximum_degree+1)*(2*maximum_degree+1)^2 complex coefficients.
   Degrees 0..64 are supported; capacity counts dcomplex elements. No allocation
   on success; padding inside each square is zero and trailing capacity is
   untouched. Failure preserves numerical output. Diagnostic storage must not
   overlap it. Returns 1 on success; no exception or process termination. */
int openmx_basis_real_harmonics_fill(int maximum_degree, dcomplex *output,
    size_t capacity, char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
