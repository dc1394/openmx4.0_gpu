#ifndef OPENMX_DENSITY_GPU_STATE_H
#define OPENMX_DENSITY_GPU_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed from the explicit calculation owner. Bind/clear/finish require
   quiescent users and completed GPU work. No device is initialized by binding.
   Geometry invalidation retains grow-only host staging; clear/finish releases
   all storage and invalidates this view. */
typedef struct OpenMXDensityGpuState OpenMXDensityGpuState;
OpenMXDensityGpuState *Set_Density_Grid_GPU_Bind_Resources(OpenMXCoreContext *core);
void Set_Density_Grid_GPU_Invalidate(OpenMXDensityGpuState *density_state);
void Set_Density_Grid_GPU_EndCycle(OpenMXDensityGpuState *density_state);
#if defined(OPENMX_GPU_HIP) && OPENMX_GPU_HIP
void Set_Density_Grid_GPU_Release_Device_Caches(OpenMXDensityGpuState *density_state);
#endif
#ifdef __cplusplus
}
#endif
#endif
