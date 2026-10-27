#pragma once
#include "openmx_hamiltonian_cache.h"
#include "openmx_core_native_resources.h"
// Solver lifetime tests model this borrowed view. The packing fixture compiles
// the complete production cache with the real C++ owner and exercises its data.
struct OpenMXHamiltonianCache { int unused; };
static void hamiltonian_fixture_reset(OpenMXNativeResources*, void*) {}
OpenMXHamiltonianCache *Set_Hamiltonian_Cache_Bind_Resources(OpenMXCoreContext *core) {
  OpenMXNativeResources *owner=nullptr;void *metadata=nullptr;char error[256];
  if(!openmx_core_native_resources_state(core,"Set_Hamiltonian_Packed",
      sizeof(OpenMXHamiltonianCache),hamiltonian_fixture_reset,&owner,&metadata,error,sizeof(error)))
    std::abort();
  return static_cast<OpenMXHamiltonianCache*>(metadata);
}
