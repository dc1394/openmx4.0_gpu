#ifndef OPENMX_BASIS_QUADRATURE_H
#define OPENMX_BASIS_QUADRATURE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Generate count nodes and weights on [-1,1], in increasing node order.
   Output arrays must each have count elements and must not overlap.
   Returns 1 on success, 0 on failure; failure leaves both arrays unchanged.
   No process termination, runtime initialization or retained pointers. */
int openmx_basis_gauss_legendre(int count, double *nodes, double *weights,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
