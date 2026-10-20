#ifndef OPENMX_CORE_ATOMIC_H
#define OPENMX_CORE_ATOMIC_H

#include "openmx_core_reciprocal.h"
#include "openmx_core_species.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Flat layout_right views of calculation-owned real-space tables. Physical
   mesh lengths are in species; extents retain capacity and interpolation
   padding. Density, density_backup, Hartree and PCC have the legacy +2 slots.
   Mutable views let the reader/normalization fill the same values consumed by
   the integrals, with no duplicate table or process-global lookup. */
typedef struct {
  OpenMXSpeciesViews species;
  OpenMXRadialView2 pao_log_radii, pao_radii, density, density_backup;
  OpenMXRadialView4 orbitals;
  OpenMXRadialView2 vps_log_radii, vps_radii, neutral_potential, hartree_potential, partial_core_density;
  OpenMXRadialView4 projectors;
  OpenMXRadialView3 projector_energies;
} OpenMXAtomicViews;

/* No allocation or Kokkos initialization. Each absent owner is an empty view.
   Failure preserves output. Expiry follows the respective owner's release,
   finish or destruction; allocation of another owner does not refresh a copy. */
int openmx_core_atomic_get(OpenMXCoreContext *context, OpenMXAtomicViews *views,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
