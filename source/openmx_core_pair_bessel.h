#ifndef OPENMX_CORE_PAIR_BESSEL_H
#define OPENMX_CORE_PAIR_BESSEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpenMXPairBesselWorkspace OpenMXPairBesselWorkspace;
typedef struct {
  const double *values;
  const double *derivatives;
  size_t points;
  size_t angular_rows;
} OpenMXPairBesselTable;

/* Copies and validates the grid. Each caller/thread owns its workspace. */
OpenMXPairBesselWorkspace *openmx_core_pair_bessel_create(int maximum_angular,
    int points, const double *wavevectors, size_t cache_limit, char *error, size_t capacity);
/* Read-only tables have angular_rows rows of points values. Cached views
   survive further samples until destruction; uncached views survive until
   the next sample. Failure leaves table unchanged. */
int openmx_core_pair_bessel_sample(OpenMXPairBesselWorkspace *workspace,
    double distance, int angular, OpenMXPairBesselTable *table, char *error, size_t capacity);
void openmx_core_pair_bessel_statistics(const OpenMXPairBesselWorkspace *workspace,
    long *hits, long *misses);
void openmx_core_pair_bessel_destroy(OpenMXPairBesselWorkspace *workspace);

#ifdef __cplusplus
}
#endif

#endif
