#ifndef OPENMX_CORE_ORBITAL_WORKSPACE_H
#define OPENMX_CORE_ORBITAL_WORKSPACE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpenMXOrbitalWorkspace OpenMXOrbitalWorkspace;
typedef struct { int maximum_angular, multiplicity, derivatives; } OpenMXOrbitalShape;

/* Flat layout_right rows. RF/dRF: [angular_channels][multiplicity];
   AF/dAFQ/dAFP: [angular_channels][magnetic_capacity]. Derivative channels
   are null in value-only workspaces. All values initially zero. Views expire
   at destruction; they may be reused only by the owning evaluation/worker. */
typedef struct {
  size_t angular_channels, multiplicity, magnetic_capacity;
  double *RF, *dRF, *AF, *dAFQ, *dAFP;
} OpenMXOrbitalBuffers;

/* No runtime initialization. Failure preserves buffers, and no C++ exception
   crosses the boundary. Each OpenMP worker owns a separate handle. */
OpenMXOrbitalWorkspace *openmx_core_orbital_create(const OpenMXOrbitalShape *shape,
    OpenMXOrbitalBuffers *buffers, char *error, size_t error_capacity);
void openmx_core_orbital_destroy(OpenMXOrbitalWorkspace *workspace);

#ifdef __cplusplus
}
#endif
#endif
