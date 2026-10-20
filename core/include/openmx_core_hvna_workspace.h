#ifndef OPENMX_CORE_HVNA_WORKSPACE_H
#define OPENMX_CORE_HVNA_WORKSPACE_H
#include "openmx_core_vna_preparation.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXHvnaProjectors OpenMXHvnaProjectors;
typedef struct OpenMXHvnaWorkspace OpenMXHvnaWorkspace;
typedef struct {
  int species, groups, components;
  const int *orbitals, *lengths;
  const double *energies; /* [species, group] */
} OpenMXHvnaProjectorBuffers;
typedef struct {
  size_t orbital_capacity;
  double *matrix; /* [orbital_capacity, orbital_capacity], row-major */
} OpenMXHvnaWorkspaceBuffers;

/* Copies all metadata and is independent of the preparation/input lifetime.
   A zero-group table accepts null energies. Views expire at destruction. */
OpenMXHvnaProjectors *openmx_core_hvna_projectors_create(const OpenMXVnaPreparation *preparation,
    double ***energies, OpenMXHvnaProjectorBuffers *buffers, char *error, size_t capacity);
void openmx_core_hvna_projectors_destroy(OpenMXHvnaProjectors *projectors);

/* Per-worker storage, independent of the projector owner's lifetime. Empty
   bases have a null matrix. Creation needs no GPU runtime. Failure leaves
   existing buffers unchanged, and exceptions never cross C. */
OpenMXHvnaWorkspace *openmx_core_hvna_workspace_create(const OpenMXHvnaProjectors *projectors,
    OpenMXHvnaWorkspaceBuffers *buffers, char *error, size_t capacity);
void openmx_core_hvna_workspace_destroy(OpenMXHvnaWorkspace *workspace);
#ifdef __cplusplus
}
#endif
#endif
