#ifndef OPENMX_ORBITAL_GPU_STATE_H
#define OPENMX_ORBITAL_GPU_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Borrowed from the explicit calculation. Geometry invalidation releases its
   orbital tables; clear/finish/destroy also invalidate this view. All users
   must be quiescent at these boundaries. Binding does not initialize a GPU. */
typedef struct OpenMXOrbitalGpuState OpenMXOrbitalGpuState;
OpenMXOrbitalGpuState *Set_Orbitals_Grid_GPU_Bind_Resources(OpenMXCoreContext *core);
void Set_Orbitals_Grid_GPU_Release(OpenMXOrbitalGpuState *orbital_state);

#ifdef __cplusplus
}
#endif
#endif
