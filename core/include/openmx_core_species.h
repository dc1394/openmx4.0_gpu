#ifndef OPENMX_CORE_SPECIES_H
#define OPENMX_CORE_SPECIES_H

#include "openmx_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int species, angular_channels, text_capacity; } OpenMXSpeciesShape;

/* Borrowed, writable input tables. All scalar fields have species entries;
   basis/contracted_basis are contiguous [species][angular_channels]. Row
   adapters are only for unmigrated C consumers. Never free or rebind a row.
   Initial mass is -1; all other values and label bytes start at zero. */
typedef struct {
  size_t species, angular_channels, text_capacity;
  int *basis_lmax, *open_core, *species_index, *relativistic;
  int *vps_mesh, *pao_mesh, *projector_components, *radial_projectors;
  int *pao_lmax, *pao_multiplicity, *atomic_number, *orbitals, *contracted_orbitals;
  double *mass, *electrons, *cutoff, *core_charge;
  int *basis, *contracted_basis;
  int **basis_rows, **contracted_basis_rows;
  char **name, **basis_name, **basis_file, **potential_file;
} OpenMXSpeciesViews;

/* Allocation is transactional and requires an unfinished, empty species owner.
   Get does not allocate or initialize Kokkos. Missing storage returns empty
   views. Failed calls preserve output. Views expire at release/finish/destroy. */
int openmx_core_species_allocate(OpenMXCoreContext *context, const OpenMXSpeciesShape *shape,
    OpenMXSpeciesViews *views, char *error, size_t error_capacity);
int openmx_core_species_get(OpenMXCoreContext *context, OpenMXSpeciesViews *views,
    char *error, size_t error_capacity);
void openmx_core_species_release(OpenMXCoreContext *context);

#ifdef __cplusplus
}
#endif
#endif
