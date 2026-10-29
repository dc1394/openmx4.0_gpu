#ifndef OPENMX_HAMILTONIAN_GPU_STATE_H
#define OPENMX_HAMILTONIAN_GPU_STATE_H
#include "openmx_core.h"
#include "openmx_core_atomic.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed from an explicit calculation. Binding requires no GPU. The
   primary device is fixed while this state is live. Geometry invalidation
   drops matrix/OTF tables and plans, retaining reusable CUDA host staging.
   Quiescent clear/finish/destroy releases all storage and invalidates views. */
typedef struct OpenMXHamiltonianGpuState OpenMXHamiltonianGpuState;
OpenMXHamiltonianGpuState *Set_Hamiltonian_GPU_Bind_Resources(OpenMXCoreContext *core);
#if defined(OPENMX_GPU_HIP) && OPENMX_GPU_HIP
int Set_Density_Grid_OTF_Possible(OpenMXHamiltonianGpuState *hamiltonian_state,
    const OpenMXAtomicViews *atomic, int Cnt_kind);
int Set_Density_Grid_OTF_Prepare(OpenMXHamiltonianGpuState *hamiltonian_state,
    const OpenMXAtomicViews *atomic, int Cnt_kind, size_t *need_bytes, size_t *available_bytes);
int Set_Density_Grid_OTF_Run(OpenMXHamiltonianGpuState *hamiltonian_state,
    const OpenMXAtomicViews *atomic, int Cnt_kind, double *****CDM, double ***Tmp_Den_Grid);
#endif
#ifdef __cplusplus
}
#endif
#endif
