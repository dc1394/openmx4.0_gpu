#ifndef OPENMX_DC_STATE_H
#define OPENMX_DC_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed calculation state. Bind, clear and finish require quiescent
   users. Teardown is collective where GPU groups use MPI communicators
   and must run before MPI_Finalize. Clear/finish invalidates this view. */
typedef struct OpenMXDCState OpenMXDCState;
OpenMXDCState *Divide_Conquer_Bind_Resources(OpenMXCoreContext *core);
void Divide_Conquer_Release_GPU_SCache(OpenMXDCState *state);
#ifdef __cplusplus
}
#endif
#endif
