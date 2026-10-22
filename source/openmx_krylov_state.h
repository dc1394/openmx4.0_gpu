#ifndef OPENMX_KRYLOV_STATE_H
#define OPENMX_KRYLOV_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed calculation state. Bind, clear and finish require quiescent
   users. Teardown is collective where GPU groups use MPI communicators
   and must run before MPI_Finalize. Clear/finish invalidates this view. */
typedef struct OpenMXKrylovState OpenMXKrylovState;
OpenMXKrylovState *Krylov_Bind_Resources(OpenMXCoreContext *core);
void Krylov_Release_GPU_KUCache(OpenMXKrylovState *state);
#ifdef __cplusplus
}
#endif
#endif
