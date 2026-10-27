#ifndef OPENMX_HAMILTONIAN_GPU_STATE_H
#define OPENMX_HAMILTONIAN_GPU_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed from an explicit calculation. Binding requires no GPU. The
   primary device is fixed while this state is live. Geometry invalidation
   drops matrix/OTF tables and plans, retaining reusable CUDA host staging.
   Quiescent clear/finish/destroy releases all storage and invalidates views. */
typedef struct OpenMXHamiltonianGpuState OpenMXHamiltonianGpuState;
OpenMXHamiltonianGpuState *Set_Hamiltonian_GPU_Bind_Resources(OpenMXCoreContext *core);
#ifdef __cplusplus
}
#endif
#endif
