#ifndef OPENMX_DCLNO_STATE_H
#define OPENMX_DCLNO_STATE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Borrowed calculation state. Bind, clear and finish require quiescent
   users. Teardown is collective where GPU groups use MPI communicators
   and must run before MPI_Finalize. Clear/finish invalidates this view. */
typedef struct OpenMXDCLNOState OpenMXDCLNOState;
OpenMXDCLNOState *Divide_Conquer_LNO_Bind_Resources(OpenMXCoreContext *core);
#ifdef __cplusplus
}
#endif
#endif
