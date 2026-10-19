#ifndef OPENMX_CORE_VNA_PREPARATION_H
#define OPENMX_CORE_VNA_PREPARATION_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXVnaPreparation OpenMXVnaPreparation;
typedef struct {
  int maximum_angular, multiplicity, projector_angular, projector_multiplicity, points, species;
  const int *species_angular;
  int *const *basis_counts;
} OpenMXVnaPreparationShape;

/* Metadata is copied and read-only. All views survive until destruction;
   numeric rows exist only for active radial functions, without padding. */
typedef struct {
  int species, angular_channels, points, projectors, projector_components;
  int basis_lmax, basis_multiplicity, projector_lmax, projector_multiplicity;
  const int *maximum_angular, *basis_counts, *orbitals;
  const int *projector_angular, *projector_radial;
  const size_t *radial_offsets;
  size_t radial_count, radial_elements;
  double *weighted[2];
} OpenMXVnaPreparationBuffers;

/* Unchecked active indices only; offset is measured in doubles. */
static inline size_t openmx_core_vna_radial_offset(const OpenMXVnaPreparationBuffers *buffers,
    int species, int angular, int radial)
{
  return (buffers->radial_offsets[(size_t)species*buffers->angular_channels+angular]+radial)*buffers->points;
}

/* Failure leaves buffers unchanged; no exception crosses C. */
OpenMXVnaPreparation *openmx_core_vna_preparation_create(const OpenMXVnaPreparationShape *shape,
    OpenMXVnaPreparationBuffers *buffers, char *error, size_t capacity);
/* Null is accepted. */
void openmx_core_vna_preparation_destroy(OpenMXVnaPreparation *preparation);
#ifdef __cplusplus
}
#endif
#endif
