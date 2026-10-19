#ifndef OPENMX_CORE_DSVNA_H
#define OPENMX_CORE_DSVNA_H
#include "openmx_complex.h"
#include "openmx_core_orbitals.h"
#include "openmx_core_vna_preparation.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXDsvnaWorkspace OpenMXDsvnaWorkspace;
typedef struct {
  int points, projectors;
  size_t orbital_capacity, projector_capacity, radial_capacity;
  size_t conversion_rows, conversion_columns, bessel_orders, scratch_capacity;
  dcomplex *angular[4], *conversion[4];
  double *sums[2], *radial[2], *bessel[2], *scratch[2];
} OpenMXDsvnaBuffers;

/* Active indices only. Magnetic indices are shifted from -l..l to 0..2*l. */
static inline size_t openmx_core_dsvna_pair(const OpenMXDsvnaBuffers *buffers,
    const OpenMXVnaPreparationBuffers *preparation, int species, int l, int radial,
    int magnetic, int projector, int projector_magnetic)
{
  return openmx_core_vna_basis_orbital(preparation,species,l,radial,magnetic)*buffers->projector_capacity
       +openmx_core_orbital_index(preparation->projector_multiplicity,
           preparation->projector_angular[projector],preparation->projector_radial[projector],projector_magnetic);
}
static inline size_t openmx_core_dsvna_sum(const OpenMXDsvnaBuffers *buffers,
    const OpenMXVnaPreparationBuffers *preparation, int species, int l, int radial, int projector)
{
  return openmx_core_vna_basis_radial(preparation,species,l,radial)*buffers->projectors+projector;
}

/* Copies capacities only. All views expire at destruction; empty angular
   and sum buffers are null. Failure leaves buffers unchanged. No GPU init. */
OpenMXDsvnaWorkspace *openmx_core_dsvna_create(const OpenMXVnaPreparation *preparation,
    OpenMXDsvnaBuffers *buffers, char *error, size_t capacity);
void openmx_core_dsvna_destroy(OpenMXDsvnaWorkspace *workspace);
#ifdef __cplusplus
}
#endif
#endif
