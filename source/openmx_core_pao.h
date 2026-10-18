#ifndef OPENMX_CORE_PAO_H
#define OPENMX_CORE_PAO_H

#include "openmx_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int species, angular_channels, multiplicity, radial_points, reciprocal_points;
} OpenMXPaoShape;

/* Transitional non-owning views for the existing C readers/consumers.
   Do not free or rebind any row. Valid until pao_release, core_finish, or the
   owning context is destroyed. Each numeric array has contiguous storage. */
typedef struct {
  double **log_radii, **radii;             /* [species][radial_points] */
  double **density, **density_backup;      /* [species][radial_points+2] */
  double ****orbitals, ****transforms;     /* [species][angular][multiplicity][r or k] */
} OpenMXPaoViews;

/* Requires a non-null, unfinished context with no active PAO storage. On failure neither
   the context nor views is changed. All numeric storage starts at zero. */
int openmx_core_pao_allocate(OpenMXCoreContext *context, const OpenMXPaoShape *shape,
    OpenMXPaoViews *views, char *error, size_t error_capacity);
/* Idempotent. Releases only this calculation's PAO data, not Kokkos. */
void openmx_core_pao_release(OpenMXCoreContext *context);

#ifdef __cplusplus
}
#endif

#endif
