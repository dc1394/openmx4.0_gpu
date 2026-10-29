#ifndef OPENMX_BASIS_LEBEDEV_H
#define OPENMX_BASIS_LEBEDEV_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Borrowed contiguous [point][x,y,z,weight] data on the unit sphere.
   Weights sum to 1: multiply the weighted sum by 4*pi for a surface integral.
   The producer defines the lifetime. Do not modify the borrowed coefficients. */
typedef struct {
  size_t count;
  const double *xyzw;
} OpenMXLebedevRuleData;

/* Unchecked access: point < rule->count and valid borrowed storage required. */
static inline const double *openmx_lebedev_point(const OpenMXLebedevRuleData *rule,
    size_t point)
{
  return rule->xyzw+4*point;
}

/* Fill an existing array without allocation or application initialization.
   Supported counts: 590, 1202, 2030. capacity is in doubles, at least 4*count.
   Preserves the inherited point order and decimal coefficients. Returns 1 on
   success; failure leaves all numerical output unchanged. The diagnostic
   buffer must not overlap the numerical output. Trailing capacity
   is untouched. No exception or process termination crosses this boundary. */
int openmx_basis_lebedev_fill(int count, double *xyzw, size_t capacity,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
