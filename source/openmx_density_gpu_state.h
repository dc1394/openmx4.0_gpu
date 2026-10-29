#ifndef OPENMX_DENSITY_GPU_STATE_H
#define OPENMX_DENSITY_GPU_STATE_H
#include "openmx_core.h"
#include "openmx_core_atomic.h"
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
/* OTF density borrows this calculation's Hamiltonian state internally. */
int Set_Density_Grid_GPU_OTF_Prepare(OpenMXDensityGpuState *density_state,
    const OpenMXAtomicViews *atomic, int Cnt_kind, size_t *need_bytes, size_t *available_bytes);
int Set_Density_Grid_GPU_OTF_Run(OpenMXDensityGpuState *density_state,
    const OpenMXAtomicViews *atomic, int Cnt_kind, double *****CDM, double ***Tmp_Den_Grid);
#endif
#ifdef __cplusplus
}
#endif
#endif
