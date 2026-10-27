#pragma once
#include "openmx_density_gpu_state.h"
#include "openmx_core_native_resources.h"
// Solver-only fixtures model the borrowed density state. The separate density
// lifecycle fixture exercises its production layout, constructor and teardown.
#if OPENMX_GPU_HIP
struct OpenMXDensityGpuState { int unused; };
static void density_fixture_reset(OpenMXNativeResources*, void*) {}
OpenMXDensityGpuState *Set_Density_Grid_GPU_Bind_Resources(OpenMXCoreContext *core) {
  OpenMXNativeResources *owner=nullptr;void *metadata=nullptr;char error[256];
  if(!openmx_core_native_resources_state(core,"Set_Density_Grid_GPU",
      sizeof(OpenMXDensityGpuState),density_fixture_reset,&owner,&metadata,error,sizeof(error)))
    std::abort();
  return static_cast<OpenMXDensityGpuState*>(metadata);
}
#endif
