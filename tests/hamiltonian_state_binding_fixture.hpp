#pragma once
#include "openmx_hamiltonian_gpu_state.h"
#include "openmx_core_native_resources.h"
// Borrowed-view fixture. The Hamiltonian lifecycle test exercises production
// tables, plans, allocation failures and teardown with the real core owner.
struct OpenMXHamiltonianGpuState { int unused; };
static void hamiltonian_gpu_fixture_reset(OpenMXNativeResources*, void*) {}
OpenMXHamiltonianGpuState *Set_Hamiltonian_GPU_Bind_Resources(OpenMXCoreContext *core) {
  OpenMXNativeResources *owner=nullptr;void *metadata=nullptr;char error[256];
  if(!openmx_core_native_resources_state(core,"Set_Hamiltonian_GPU",
      sizeof(OpenMXHamiltonianGpuState),hamiltonian_gpu_fixture_reset,&owner,&metadata,error,sizeof(error)))
    std::abort();
  return static_cast<OpenMXHamiltonianGpuState*>(metadata);
}
