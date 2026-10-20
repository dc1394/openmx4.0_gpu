#ifndef OPENMX_CORE_VPS_H
#define OPENMX_CORE_VPS_H

#include "openmx_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int species, spins, projectors, radial_points, reciprocal_points;
} OpenMXVpsShape;

/* Species and radial capacity are inherited from OpenMXVpsShape. */
typedef struct {
  int angular_channels, multiplicity, wavevectors;
} OpenMXNeutralProjectorShape;

/* Borrowed C views, valid until vps_release, core_finish or context destruction.
   Do not free or rebind rows. Every numeric array is contiguous and initialized
   to zero, including inactive channels and padding. All dimensions are capacities. */
typedef struct {
  double **log_radii, **radii, **neutral_potential; /* [species][radial_points] */
  double **hartree_potential, **partial_core_density; /* [species][radial_points+2] */
  double ****projectors;             /* [spins][species][projectors][radial_points] */
  double ***energies;                /* [spins][species][projectors] */
  int **angular;                    /* [species][projectors], legacy list with index 0 */
  double ****transforms;             /* [spins][species][projectors+2][reciprocal_points] */
  /* Null when neutral_shape is null. */
  double ****neutral_projectors;     /* [species][angular][multiplicity][radial_points] */
  double ***neutral_energies;        /* [species][angular][multiplicity] */
  double ****neutral_transforms;     /* [species][angular][multiplicity][wavevectors+2] */
  double **potential_transform;      /* [species][wavevectors+2] */
} OpenMXVpsViews;

/* Requires an unfinished context with no VPS storage. PAO storage is independent.
   neutral_shape may be null to disable VNA expansion. Failure preserves context
   and output views, including failure after some numeric arrays were allocated. */
int openmx_core_vps_allocate(OpenMXCoreContext *context, const OpenMXVpsShape *shape,
    const OpenMXNeutralProjectorShape *neutral_shape, OpenMXVpsViews *views,
    char *error, size_t error_capacity);
/* Idempotent. Does not release PAO storage or the Kokkos runtime. */
void openmx_core_vps_release(OpenMXCoreContext *context);

#ifdef __cplusplus
}
#endif

#endif
