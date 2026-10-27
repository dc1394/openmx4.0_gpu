#pragma once
#include "openmx_orbital_gpu_state.h"
#include "openmx_core_native_resources.h"
// Borrowed-view fixture only; the orbital lifecycle test compiles production
// construction, upload, launch and cleanup against the actual core owner.
struct OpenMXOrbitalGpuState { int unused; };
static void orbital_fixture_reset(OpenMXNativeResources*, void*) {}
OpenMXOrbitalGpuState *Set_Orbitals_Grid_GPU_Bind_Resources(OpenMXCoreContext *core) {
  OpenMXNativeResources *owner=nullptr;void *metadata=nullptr;char error[256];
  if(!openmx_core_native_resources_state(core,"Set_Orbitals_Grid_GPU",
      sizeof(OpenMXOrbitalGpuState),orbital_fixture_reset,&owner,&metadata,error,sizeof(error)))
    std::abort();
  return static_cast<OpenMXOrbitalGpuState*>(metadata);
}
