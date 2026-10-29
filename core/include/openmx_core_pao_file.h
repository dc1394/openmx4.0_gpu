#ifndef OPENMX_CORE_PAO_FILE_H
#define OPENMX_CORE_PAO_FILE_H

#include "openmx_core_atomic.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double atom_species, cutoff;
  int atomic_number, mesh_points, lmax, multiplicity;
  int eigenvalue_found;
  double eigenvalue;
} OpenMXPaoFileInfo;

/* Adapt the independent basis reader to calculation-owned flat tables.
   metadata_only must be 0 or 1. A metadata-only call updates species fields
   but needs no PAO numerical allocation. Full reads also validate the selected
   basis and every output extent, then copy physical values (density at +1).
   Padding and other species remain unchanged. eigen_angular=-1 skips lookup;
   otherwise request the optional multiplicity-zero generator energy.
   The reader owns the file and all intermediate arrays using C++ RAII.
   Failure leaves both information and every target field/buffer unchanged.
   No Kokkos, MPI or mutable process-global input state is used. */
int openmx_core_pao_read_file(const char *path, int metadata_only, int eigen_angular,
    const OpenMXAtomicViews *target, size_t species, OpenMXPaoFileInfo *information,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
